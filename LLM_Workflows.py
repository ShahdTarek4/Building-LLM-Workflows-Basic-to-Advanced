import os
import json
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
model_server = os.getenv('MODEL_SERVER', 'GROQ').upper()  # Default to GROQ if not set


if model_server == "GROQ":
    API_KEY = os.getenv('GROQ_API_KEY')
    BASE_URL = os.getenv('GROQ_BASE_URL')
    LLM_MODEL = os.getenv('GROQ_MODEL')
   
elif model_server == "NGU":
    API_KEY = os.getenv('NGU_API_KEY')
    BASE_URL = os.getenv('NGU_BASE_URL')
    LLM_MODEL = os.getenv('NGU_MODEL')

else:
    raise ValueError(f"Unsupported MODEL_SERVER: {model_server}")

# Initialize OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def call_llm(messages, tools=None, tool_choice=None):
    """Make a call to the LLM API with the specified messages and tools."""
    kwargs = {
        "model": LLM_MODEL,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice
    try:
        response = client.chat.completions.create(**kwargs)
        return response
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None

def get_sample_blog_post():
    """Read the sample blog post from a JSON file."""
    try:
        with open('sample_blog_post.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Error: sample_blog_post.json file not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in sample_blog_post.json.")
        return None


extract_key_points_schema = {
    "type": "function",
    "function": {
        "name": "extract_key_points",
        "description": "Extract key points from a blog post",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title of the blog post"},
                "content": {"type": "string", "description": "The content of the blog post"},
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of key points extracted from the blog post"
                }
            },
            "required": ["key_points"]
        }
    }
}

generate_summary_schema = {
    "type": "function",
    "function": {
        "name": "generate_summary",
        "description": "Generate a concise summary from the key points",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Concise summary of the blog post"}
            },
            "required": ["summary"]
        }
    }
}

create_social_media_posts_schema = {
    "type": "function",
    "function": {
        "name": "create_social_media_posts",
        "description": "Create social media posts for different platforms",
        "parameters": {
            "type": "object",
            "properties": {
                "twitter": {"type": "string", "description": "Post optimized for Twitter/X (max 280 characters)"},
                "linkedin": {"type": "string", "description": "Post optimized for LinkedIn (professional tone)"},
                "facebook": {"type": "string", "description": "Post optimized for Facebook"}
            },
            "required": ["twitter", "linkedin", "facebook"]
        }
        }
}

create_email_newsletter_schema = {
    "type": "function",
    "function": {
        "name": "create_email_newsletter",
        "description": "Create an email newsletter from the blog post and summary",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body content in plain text"}
            },
            "required": ["subject", "body"]
        }
    }
}

finish_tool_schema = {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Complete the workflow and return final results",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Final summary"},
                    "social_posts": {
                        "type": "object",
                        "properties": {
                            "twitter": {"type": "string"},
                            "linkedin": {"type": "string"},
                            "facebook": {"type": "string"}
                        }
                    },
                    "email": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "body": {"type": "string"}
                        }
                    }
                },
                  "required": ["summary", "social_posts", "email"]
            }
        }
    }
    
def task_extract_key_points(blog_post):
    messages = [
        {"role": "system", "content": "You are an expert at analyzing content and extracting key points from articles."},
        {"role": "user", "content": f"Extract the key points from this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"}
    ]
    response = call_llm(
        messages=messages,
        tools=[extract_key_points_schema],
        tool_choice={"type": "function", "function": {"name": "extract_key_points"}}
    )
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        return result.get("key_points", [])
    return []

def task_generate_summary(key_points, max_length=150):
    messages = [
        {"role": "system", "content": "You are an expert at summarizing content concisely while preserving key information."},
        {"role": "user", "content": f"Generate a summary based on these key points, max {max_length} words:\n\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(
        messages=messages,
        tools=[generate_summary_schema],
        tool_choice={"type": "function", "function": {"name": "generate_summary"}}
    )
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        return result.get("summary", "")
    return ""

def task_create_social_media_posts(key_points, blog_title):
    messages = [
        {"role": "system", "content": "You are a social media expert who creates engaging posts optimized for different platforms."},
        {"role": "user", "content": f"Create social media posts for Twitter, LinkedIn, and Facebook based on this blog title: '{blog_title}' and these key points:\n\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(
        messages=messages,
        tools=[create_social_media_posts_schema],
        tool_choice={"type": "function", "function": {"name": "create_social_media_posts"}}
    )
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    return {"twitter": "", "linkedin": "", "facebook": ""}


def task_create_email_newsletter(blog_post, summary, key_points):
    messages = [
        {"role": "system", "content": "You are an email marketing specialist who creates engaging newsletters."},
        {"role": "user", "content": f"Create an email newsletter based on this blog post:\n\nTitle: {blog_post['title']}\n\nSummary: {summary}\n\nKey Points:\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(
        messages=messages,
        tools=[create_email_newsletter_schema],
        tool_choice={"type": "function", "function": {"name": "create_email_newsletter"}}
    )
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    return {"subject": "", "body": ""}


def run_pipeline_workflow(blog_post):
    """Run tasks sequentially in a pipeline"""
    key_points = task_extract_key_points(blog_post)
    summary = task_generate_summary(key_points)
    social_posts = task_create_social_media_posts(key_points, blog_post['title'])
    email = task_create_email_newsletter(blog_post, summary, key_points)
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

def run_dag_workflow(blog_post):
    """Run tasks in a DAG structure with parallel execution where possible"""
    # First extract key points 
    key_points = task_extract_key_points(blog_post)
    
    # Then run these tasks in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        summary_future = executor.submit(task_generate_summary, key_points)
        social_posts_future = executor.submit(task_create_social_media_posts, key_points, blog_post['title'])
        
        summary = summary_future.result()
        social_posts = social_posts_future.result()
    
    #create email (depends on both blog_post and summary)
    email = task_create_email_newsletter(blog_post, summary, key_points)
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

def evaluate_content(content, content_type):
    """Enhanced evaluation with all required content types"""
    criteria = {
        "key_points": "1. Minimum 5 key points\n2. Each point >8 words\n3. Technical specificity",
        "summary": "1. 50-100 words\n2. Covers all key points\n3. No fluff",
        "social_media_posts": "1. Platform-specific formatting\n2. Hashtags where appropriate\n3. Clear CTA",
        "email_newsletter": "1. Professional structure\n2. Bullet points\n3. Personalization",
        "email": "1. Professional structure\n2. Bullet points\n3. Personalization"  
    }
    
    content_type = content_type.lower()
    if content_type not in criteria:
        return {"quality_score": 0.5, "feedback": f"Invalid content type: {content_type}"}
    
    messages = [
        {"role": "system", "content": f"""
        Evaluate this {content_type} (0-1 scale). Criteria:
        {criteria[content_type]}
        Return format: "score|feedback"
        """},
        {"role": "user", "content": str(content)}
    ]
    response = call_llm(messages)
    if not response:
        return {"quality_score": 0.5, "feedback": "Evaluation failed"}
    
    try:
        score, feedback = response.choices[0].message.content.split("|", 1)
        return {"quality_score": float(score), "feedback": feedback.strip()}
    except:
        return {"quality_score": 0.5, "feedback": "Invalid evaluation format"}



def generate_with_reflexion(generator_func, max_attempts=3):
    """Apply Reflexion to a content generation function.
    Args:
        generator_func: Original content generation function
        max_attempts: Maximum refinement attempts
    Returns:
        Function that generates self-corrected content
    """
    def wrapped_generator(*args, **kwargs):
        content_type = kwargs.pop("content_type", 
            generator_func.__name__.replace("task_", ""))
        
        best_score = 0
        best_content = None
        
        for attempt in range(max_attempts):
            content = generator_func(*args, **kwargs)
            evaluation = evaluate_content(content, content_type)
            
            if evaluation["quality_score"] > best_score:
                best_score = evaluation["quality_score"]
                best_content = content
            
            if best_score >= 0.8:  # Quality threshold from assignment
                break
            content = improve_content(content, evaluation["feedback"], content_type)
        
        return best_content
    
    return wrapped_generator

def improve_content(content, feedback, content_type):
    """
    Args:
        content: Content to improve
        feedback: Specific improvement suggestions
        content_type: Type of content being improved
    Returns:
        Improved content
    """
    messages = [
        {
            "role": "system",
            "content": f"""Improve this {content_type} based on feedback.
            Keep the same format but enhance quality substantially."""
        },
        {
            "role": "user",
            "content": f"""Current content:\n{content}\n\nFeedback:\n{feedback}\n
            Provide the enhanced version:"""
        }
    ]
    
    response = call_llm(messages)
    return response.choices[0].message.content if response else content   

def run_workflow_with_reflexion(blog_post):
    """Run a workflow with Reflexion-based self-correction."""
    # Create enhanced versions of all task functions
    enhanced_extract = generate_with_reflexion(task_extract_key_points)
    enhanced_summary = generate_with_reflexion(task_generate_summary)
    enhanced_social = generate_with_reflexion(task_create_social_media_posts)
    enhanced_email = generate_with_reflexion(task_create_email_newsletter)
    
    # Execute workflow with self-correction
    key_points = enhanced_extract(blog_post, content_type="key_points")
    summary = enhanced_summary(key_points, content_type="summary")
    social_posts = enhanced_social(key_points, blog_post["title"], content_type="social_media_posts")
    email = enhanced_email(blog_post, summary, key_points, content_type="email")
    
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

class Agent:
    def __init__(self, name, task_function, description):
        self.name = name
        self.task_function = task_function
        self.description = description
    
    def execute(self, *args, **kwargs):
        print(f"Agent {self.name} executing: {self.description}")
        return self.task_function(*args, **kwargs)

def run_agent_workflow(blog_post):
    """Run an agent-driven workflow to repurpose content."""
    extractor = Agent("Extractor", task_extract_key_points, "Extract key points from the blog post")
    summarizer = Agent("Summarizer", task_generate_summary, "Generate a concise summary from key points")
    social_media_creator = Agent("Social Media Creator", task_create_social_media_posts, "Create social media posts for multiple platforms")
    email_composer = Agent("Email Composer", task_create_email_newsletter, "Compose an email newsletter")
    key_points = extractor.execute(blog_post)
    if not key_points:
        return {"error": "Agent Extractor failed to extract key points"}
    summary = summarizer.execute(key_points)
    if not summary:
        return {"error": "Agent Summarizer failed to generate summary"}
    social_posts = social_media_creator.execute(key_points, blog_post['title'])
    if not social_posts["twitter"]:
        return {"error": "Agent Social Media Creator failed to create social media posts"}
    email = email_composer.execute(blog_post, summary, key_points)
    if not email["subject"]:
        return {"error": "Agent Email Composer failed to create email newsletter"}
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }


# Bonus Challenge: Comparative Evaluation System
def evaluate_workflow_output(output, workflow_name):
    """Evaluate the quality of a workflow's output."""
    if "error" in output:
        return {
            "workflow": workflow_name,
            "overall_score": 0.0,
            "components": {
                "key_points": {"score": 0.0, "feedback": "Failed to generate"},
                "summary": {"score": 0.0, "feedback": "Failed to generate"},
                "social_posts": {"score": 0.0, "feedback": "Failed to generate"},
                "email": {"score": 0.0, "feedback": "Failed to generate"}
            }
        }
    
    components = {}
    components["key_points"] = evaluate_content(output["key_points"], "key_points_list")
    components["summary"] = evaluate_content(output["summary"], "summary")
    components["social_posts"] = evaluate_content(output["social_posts"], "social_media_posts")
    components["email"] = evaluate_content(output["email"], "email_newsletter")
    
    overall_score = sum(comp["quality_score"] for comp in components.values()) / len(components)
    
    return {
        "workflow": workflow_name,
        "overall_score": overall_score,
        "components": components
    }

def compare_workflows(blog_post):
    """Run all workflows, evaluate their outputs, and compare strengths and weaknesses."""
    # Run each workflow
    pipeline_result = run_pipeline_workflow(blog_post)
    dag_result = run_dag_workflow(blog_post)
    reflexion_result = run_workflow_with_reflexion(blog_post)
    agent_result = run_agent_workflow(blog_post)
    
    # Evaluate outputs
    evaluations = [
        evaluate_workflow_output(pipeline_result, "Basic Pipeline"),
        evaluate_workflow_output(dag_result, "DAG Workflow"),
        evaluate_workflow_output(reflexion_result, "Reflexion Workflow"),
        evaluate_workflow_output(agent_result, "Agent-Driven Workflow")
    ]
    
    # Generate comparison
    messages = [
        {"role": "system", "content": "You are an expert analyst comparing different workflow approaches. Provide a detailed comparison of strengths and weaknesses based on the evaluations. Format as:\n**Workflow Name**\nStrengths: [text]\nWeaknesses: [text]"},
        {"role": "user", "content": f"Compare these workflow evaluations:\n\n{json.dumps(evaluations, indent=2)}"}
    ]
    response = call_llm(messages)
    if response and response.choices[0].message.content:
        comparison = response.choices[0].message.content.strip()
    else:
        comparison = "Failed to generate comparison"
    
    return {
        "evaluations": evaluations,
        "comparison": comparison
    }


if __name__ == "__main__":
    blog_post = get_sample_blog_post()
    if blog_post:
       #print("=== Pipeline Workflow Results ===")
       #print(json.dumps(run_pipeline_workflow(blog_post), indent=2))
       #print("\n=== DAG Workflow Results ===")
       #print(json.dumps(run_dag_workflow(blog_post), indent=2))
       #print("\n=== Workflow with Reflexion ===")
       #print(json.dumps(run_workflow_with_reflexion(blog_post), indent=2))
       #print("\n=== Agent-Driven Workflow ===")
       #print(json.dumps(run_agent_workflow(blog_post), indent=2))
       print("\n-===Comparative Evaluation System (Bonus Challenge)===")
       print(json.dumps(compare_workflows(blog_post), indent=2))
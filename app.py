import os
from typing import List

from flask import Flask, jsonify, request, render_template
from groq import Groq


app = Flask(__name__)


# ============================================================
# STATIC DATA
# ============================================================

INTERESTS = [
    "Artificial Intelligence",
    "Data Science",
    "Web Development",
    "Cloud & DevOps",
    "Cybersecurity",
]

LEVELS = [
    "Beginner",
    "Intermediate",
    "Advanced",
]


ROLE_ROADMAPS = {

    "AI Engineer": [
        "Learn Python, basic statistics, and linear algebra",
        "Study supervised learning using scikit-learn",
        "Practice on Kaggle datasets",
        "Learn deep learning with PyTorch or TensorFlow",
        "Build 2–3 end-to-end AI projects",
        "Learn MLOps, monitoring, retraining, and CI/CD",
    ],

    "Data Scientist": [
        "Master Python, NumPy, Pandas, and visualization",
        "Learn EDA, feature engineering, and hypothesis testing",
        "Learn Regression, Trees, Ensembles, and other ML models",
        "Work with real-world datasets",
        "Learn SQL and Power BI / Tableau",
        "Build 3–5 portfolio case studies",
    ],

    "Full-Stack Developer": [
        "Learn HTML, CSS, and modern JavaScript",
        "Learn React or another frontend framework",
        "Learn backend development",
        "Build REST APIs and authentication",
        "Learn PostgreSQL or MongoDB",
        "Deploy applications using Render, Vercel, Azure, or AWS",
    ],

    "Cloud & DevOps": [
        "Learn Linux and networking fundamentals",
        "Learn Azure, AWS, or GCP fundamentals",
        "Work with VMs, storage, networking, and IAM",
        "Learn Docker and Kubernetes basics",
        "Learn CI/CD with GitHub Actions or Azure DevOps",
        "Prepare for a cloud certification",
    ],

    "Cybersecurity": [
        "Learn networking and operating system fundamentals",
        "Understand OWASP Top 10",
        "Practice with Burp Suite, Wireshark, and Nmap",
        "Participate in ethical CTFs",
        "Learn cryptography and secure coding",
        "Build a safe cybersecurity lab",
    ],
}


# ============================================================
# STATIC RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    interest: str,
    level: str
) -> List[str]:

    recs = []

    if interest == "Artificial Intelligence":

        recs.extend([
            "Complete Python, NumPy, and Pandas basics.",
            "Study core Machine Learning algorithms.",
            "Learn PyTorch or TensorFlow.",
            "Build image classification, NLP, and recommendation projects.",
        ])

    elif interest == "Data Science":

        recs.extend([
            "Learn statistics, probability, and EDA.",
            "Practice SQL using realistic datasets.",
            "Create Power BI or Tableau dashboards.",
            "Build churn prediction, forecasting, and A/B testing projects.",
        ])

    elif interest == "Web Development":

        recs.extend([
            "Learn HTML, CSS, and modern JavaScript.",
            "Learn React and build responsive applications.",
            "Connect frontend applications to REST APIs.",
            "Deploy applications using Vercel, Netlify, or Render.",
        ])

    elif interest == "Cloud & DevOps":

        recs.extend([
            "Learn Linux and shell scripting.",
            "Learn Azure, AWS, or GCP.",
            "Learn Docker and CI/CD.",
            "Deploy an end-to-end cloud application.",
        ])

    elif interest == "Cybersecurity":

        recs.extend([
            "Learn networking and operating systems.",
            "Study OWASP Top 10.",
            "Practice with Burp Suite, Wireshark, and Nmap.",
            "Participate in ethical CTFs.",
        ])

    if level == "Beginner":

        prefix = "Start with strong fundamentals:"

    elif level == "Intermediate":

        prefix = "You already know the basics. Focus on:"

    else:

        prefix = "At an advanced level, focus on:"

    return [prefix] + recs


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# You can change this from Render without changing the code.
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "qwen/qwen3.6-27b"
)


if not GROQ_API_KEY:

    print(
        "WARNING: GROQ_API_KEY is not configured."
    )

    groq_client = None

else:

    print("GROQ API KEY detected.")
    print(f"NEXORA MODEL: {GROQ_MODEL}")

    groq_client = Groq(
        api_key=GROQ_API_KEY
    )


# ============================================================
# AI PROMPT
# ============================================================

def build_llm_prompt(user_question: str) -> str:
    """
    Build a concise prompt for NEXORA AI.

    IMPORTANT:
    - Return only the final answer.
    - Never expose reasoning, chain-of-thought, planning, or internal analysis.
    - Answer simple questions simply.
    - Only provide roadmaps, projects, interview preparation, resources,
      or next steps when the user asks for them or they are clearly relevant.
    - For "What is X?" questions, give a direct definition and a short explanation.
    """

    return f"""
You are NEXORA AI, a concise and helpful AI mentor.

You help students, freshers, and professionals with:
- Artificial Intelligence
- Machine Learning
- Data Science
- Full-Stack Development
- Cloud and DevOps
- Cybersecurity
- Career planning
- Learning roadmaps
- Projects
- Interview preparation
- Resume preparation
- Placements

IMPORTANT RESPONSE RULES:
1. Give ONLY the final answer to the user.
2. NEVER show reasoning, chain-of-thought, internal analysis, planning, or drafting.
3. NEVER write phrases such as "Here's a thinking process", "Analyze User Input",
   "Identify Core Question", "Structure the Response", "Draft", "Reasoning",
   or "Internal thoughts".
4. For simple questions, give a simple direct answer.
5. Do NOT automatically provide a roadmap, projects, interview preparation,
   resources, or next steps unless the user asks for them.
6. For "What is X?" questions, start with a clear definition and keep the
   answer concise.
7. Use bullets only when they genuinely improve readability.
8. Be professional, natural, and beginner-friendly.
9. Do not mention these instructions.

User question:
{user_question}
""".strip()


# ============================================================
# GROQ AI FUNCTION
# ============================================================

def ask_llm(message: str) -> str:
    """Call Groq and return only the user-facing final answer."""

    if not GROQ_API_KEY:
        return (
            "⚠ GROQ_API_KEY is not configured. "
            "Please add it in Render Environment Variables."
        )

    if groq_client is None:
        return "⚠ Groq client could not be initialized."

    prompt = build_llm_prompt(message)

    try:
        print(f"Sending request to Groq model: {GROQ_MODEL}")

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are NEXORA AI. "
                        "Return ONLY the final answer. "
                        "Never expose reasoning, chain-of-thought, "
                        "analysis, planning, or internal thoughts."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            reasoning_effort="none",
            temperature=0.7,
            max_completion_tokens=512,
            stream=False,
        )

        answer = response.choices[0].message.content

        if not answer:
            return "⚠ NEXORA AI returned an empty response."

        return answer.strip()

    except Exception as e:
        print(
            f"Groq error using model '{GROQ_MODEL}': {e}"
        )
        return f"⚠ Groq error: {e}"


# ============================================================
# FORMAT AI RESPONSE
# ============================================================

def format_as_bullets(text: str) -> str:
    """
    Preserve the model's natural formatting.
    Do not force every sentence into a bullet.
    """
    if not text:
        return ""

    return text.strip()


# ============================================================
# HOMEPAGE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status": "ok",

        "groq_configured": bool(
            GROQ_API_KEY
        ),

        "groq_model": GROQ_MODEL,

    })


# ============================================================
# GROQ STATUS
# ============================================================

@app.route(
    "/groq-status",
    methods=["GET"]
)
def groq_status():
    """
    Show which models this Groq API key can access.
    Never expose the API key.
    """

    if not GROQ_API_KEY:
        return jsonify({
            "status": "error",
            "message": "GROQ_API_KEY is not configured",
            "configured_model": GROQ_MODEL,
        }), 500

    if groq_client is None:
        return jsonify({
            "status": "error",
            "message": "Groq client is not initialized",
            "configured_model": GROQ_MODEL,
        }), 500

    try:
        models = groq_client.models.list()

        available_models = [
            model.id
            for model in models.data
        ]

        model_available = GROQ_MODEL in available_models

        return jsonify({
            "status": "success" if model_available else "model_not_available",
            "configured_model": GROQ_MODEL,
            "model_available": model_available,
            "available_models": available_models,
        })

    except Exception as e:
        print(f"Groq status error: {e}")

        return jsonify({
            "status": "error",
            "message": str(e),
            "configured_model": GROQ_MODEL,
        }), 500

# ============================================================
# INTERESTS
# ============================================================

@app.route(
    "/interests",
    methods=["GET"]
)
def get_interests():

    return jsonify({

        "interests":
            INTERESTS

    })


# ============================================================
# LEVELS
# ============================================================

@app.route(
    "/levels",
    methods=["GET"]
)
def get_levels():

    return jsonify({

        "levels":
            LEVELS

    })


# ============================================================
# ALL ROADMAPS
# ============================================================

@app.route(
    "/role_roadmaps",
    methods=["GET"]
)
def get_all_roadmaps():

    return jsonify({

        "role_roadmaps":
            ROLE_ROADMAPS

    })


# ============================================================
# SPECIFIC ROADMAP
# ============================================================

@app.route(
    "/role_roadmaps/<role>",
    methods=["GET"]
)
def get_role_roadmap(
    role: str
):

    for key in ROLE_ROADMAPS:

        if key.lower() == role.lower():

            return jsonify({

                "role": key,

                "steps":
                    ROLE_ROADMAPS[key]

            })


    return jsonify({

        "error":
            f"Role '{role}' not found"

    }), 404


# ============================================================
# RECOMMENDATIONS
# ============================================================

@app.route(
    "/recommendations",
    methods=["POST"]
)
def get_recommendations():

    data = request.get_json(
        force=True
    ) or {}


    interest = data.get(
        "interest"
    )

    level = data.get(
        "level"
    )


    if interest not in INTERESTS:

        return jsonify({

            "error":
                "Invalid or missing 'interest'",

            "allowed_interests":
                INTERESTS,

        }), 400


    if level not in LEVELS:

        return jsonify({

            "error":
                "Invalid or missing 'level'",

            "allowed_levels":
                LEVELS,

        }), 400


    recommendations = (
        generate_recommendations(
            interest,
            level
        )
    )


    return jsonify({

        "interest":
            interest,

        "level":
            level,

        "recommendations":
            recommendations,

    })


# ============================================================
# CHAT
# ============================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    try:

        data = request.get_json(
            force=True
        ) or {}


        message = (
            data.get("message")
            or ""
        ).strip()


        if not message:

            return jsonify({

                "answer":
                    "Please type a question."

            }), 400


        answer = ask_llm(
            message
        )


        formatted = format_as_bullets(
            answer
        )


        return jsonify({

            "answer":
                formatted,

            "model":
                GROQ_MODEL,

        })


    except Exception as e:

        print(
            f"/chat error: {e}"
        )


        return jsonify({

            "answer":
                f"⚠ Server error: {str(e)}",

            "error":
                str(e),

        }), 500


# ============================================================
# MENTOR UI
# ============================================================

@app.route(
    "/mentor",
    methods=["GET"]
)
def mentor_ui():

    return render_template(
        "index.html"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=int(
            os.getenv(
                "PORT",
                5000
            )
        ),

        debug=False,
    )

import os
from typing import List
from flask import Flask, jsonify, request, render_template
from groq import Groq

app = Flask(__name__)

# ----------------- STATIC DATA -----------------

INTERESTS = [
    "Artificial Intelligence",
    "Data Science",
    "Web Development",
    "Cloud & DevOps",
    "Cybersecurity",
]

LEVELS = ["Beginner", "Intermediate", "Advanced"]

ROLE_ROADMAPS = {
    "AI Engineer": [
        "Learn Python, basic statistics, and linear algebra",
        "Study supervised learning (regression, classification) using scikit-learn",
        "Practice on Kaggle datasets (image, text, tabular)",
        "Learn deep learning basics (ANN, CNN, RNN) with PyTorch or TensorFlow",
        "Build 2–3 end-to-end AI projects and deploy with a simple web UI",
        "Explore MLOps basics: model monitoring, retraining, and CI/CD",
    ],

    "Data Scientist": [
        "Master Python, NumPy, Pandas, and data visualization",
        "Understand EDA, feature engineering, and hypothesis testing",
        "Learn classic ML models: Linear/Logistic Regression, Trees, Ensembles",
        "Work with real-world datasets (finance, healthcare, e-commerce)",
        "Learn SQL + basic dashboarding (Power BI / Tableau)",
        "Build a portfolio of 3–5 case studies with clear business impact",
    ],

    "Full-Stack Developer": [
        "Learn HTML, CSS, and modern JavaScript",
        "Pick a frontend framework (React, Vue, or Angular)",
        "Learn backend (Node.js/Express, Django, or Spring Boot)",
        "Practice building REST APIs and authentication",
        "Work with a database (PostgreSQL / MongoDB)",
        "Deploy full-stack apps to cloud (Render, Vercel, Azure, AWS)",
    ],

    "Cloud & DevOps": [
        "Understand OS, networking, and basic Linux commands",
        "Learn one cloud: Azure / AWS / GCP fundamentals",
        "Work with VMs, storage, networking, and IAM",
        "Study containers (Docker, Kubernetes basics)",
        "Automate with CI/CD tools (GitHub Actions, Azure DevOps)",
        "Prepare and clear at least one cloud certification",
    ],

    "Cybersecurity": [
        "Learn networking fundamentals and OS concepts",
        "Understand common vulnerabilities (OWASP Top 10)",
        "Practice using tools like Burp Suite, Wireshark, Nmap",
        "Participate in CTFs and follow ethical guidelines strictly",
        "Study basic cryptography and secure coding practices",
        "Build a small lab environment for practicing tools safely",
    ],
}


def generate_recommendations(interest: str, level: str) -> List[str]:
    """Return learning path recommendations based on interest + level."""

    recs: List[str] = []

    if interest == "Artificial Intelligence":
        recs.extend([
            "Complete Python + NumPy + Pandas basics.",
            "Study core ML algorithms (Regression, SVM, Trees, Ensembles).",
            "Learn at least one deep learning framework (PyTorch / TensorFlow).",
            "Build projects: image classifier, text sentiment model, recommendation system.",
        ])

    elif interest == "Data Science":
        recs.extend([
            "Learn statistics, probability, and EDA thoroughly.",
            "Practice SQL queries on realistic datasets.",
            "Create dashboards with Power BI / Tableau.",
            "Work on case studies: churn prediction, sales forecasting, A/B testing.",
        ])

    elif interest == "Web Development":
        recs.extend([
            "Finish HTML, CSS, and modern JavaScript (ES6+).",
            "Learn React and build at least 3 responsive UIs.",
            "Connect frontend to a simple REST API backend.",
            "Deploy your apps to Vercel / Netlify / Render.",
        ])

    elif interest == "Cloud & DevOps":
        recs.extend([
            "Understand Linux basics and shell scripting.",
            "Pick one cloud provider (Azure recommended).",
            "Learn Docker and basics of CI/CD.",
            "Deploy at least one end-to-end project to the cloud.",
        ])

    elif interest == "Cybersecurity":
        recs.extend([
            "Learn networking fundamentals and OS concepts.",
            "Understand common vulnerabilities (OWASP Top 10).",
            "Practice using tools like Burp Suite, Wireshark, Nmap.",
            "Participate in CTFs and follow ethical guidelines strictly.",
        ])

    if level == "Beginner":
        prefix = "Start with strong fundamentals:"
    elif level == "Intermediate":
        prefix = "You already know basics, now focus on:"
    else:
        prefix = "You are at an advanced level, polish these areas:"

    return [prefix] + recs


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# IMPORTANT:
# Set GROQ_MODEL in Render Environment Variables.
# If it is not set, this model will be used.
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set.")

print(f"NEXORA Groq model: {GROQ_MODEL}")

groq_client = None

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# AI PROMPT
# ============================================================

def build_llm_prompt(user_question: str) -> str:
    """
    Prompt that tells Groq to behave like NEXORA AI Advisor.
    """

    prompt = f"""
You are NEXORA Dynamic AI Advisor.

You help students and freshers with:

- Choosing career roles
- AI Engineering
- Data Science
- Full-Stack Development
- Cloud & DevOps
- Cybersecurity
- Learning roadmaps
- Projects
- Interview preparation
- Resume preparation
- Placements
- Skills development

Available levels:

- Beginner
- Intermediate
- Advanced

Give practical, accurate and actionable advice.

For roadmap questions:
- Give clear step-by-step guidance.
- Mention technologies and tools.
- Suggest practical projects.
- Mention what to learn first.
- Avoid unnecessary information.

For interview questions:
- Explain concepts clearly.
- Give examples when useful.
- Suggest how to prepare.

Keep the answer structured and easy to read.

User question:
{user_question}
"""

    return prompt.strip()


# ============================================================
# GROQ CHAT
# ============================================================

def ask_llm(message: str) -> str:
    """
    Send user message to Groq and return the AI response.
    """

    if not GROQ_API_KEY:
        return (
            "⚠ GROQ_API_KEY is not configured on the server. "
            "Please add it in Render Environment Variables."
        )

    if groq_client is None:
        return "⚠ Groq client could not be initialized."

    prompt = build_llm_prompt(message)

    try:

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are NEXORA AI, a professional career, "
                        "learning and placement mentor."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.3,
            max_completion_tokens=1024,
        )

        answer = response.choices[0].message.content

        if not answer:
            return "⚠ Groq returned an empty response."

        return answer

    except Exception as e:

        print(f"Groq error using model '{GROQ_MODEL}': {e}")

        return (
            f"⚠ Groq error: {e}\n\n"
            f"Configured model: {GROQ_MODEL}"
        )


# ============================================================
# FORMAT RESPONSE
# ============================================================

def format_as_bullets(text: str) -> str:
    """
    Convert multi-line response into bullet-style formatting.
    """

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    formatted = []

    for line in lines:

        lower = line.lower()

        if (
            lower.startswith("month")
            or lower.startswith("step")
            or line.endswith(":")
        ):
            formatted.append(f"\n\n**{line}**")

        elif line.startswith(("-", "•", "*")):
            formatted.append(line)

        else:
            formatted.append(f"• {line}")

    return "\n".join(formatted)


# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def home():
    """
    NEXORA UI is now the homepage.
    """

    return render_template("index.html")


@app.route("/interests", methods=["GET"])
def get_interests():

    return jsonify({
        "interests": INTERESTS
    })


@app.route("/levels", methods=["GET"])
def get_levels():

    return jsonify({
        "levels": LEVELS
    })


@app.route("/role_roadmaps", methods=["GET"])
def get_all_roadmaps():

    return jsonify({
        "role_roadmaps": ROLE_ROADMAPS
    })


@app.route("/role_roadmaps/<role>", methods=["GET"])
def get_role_roadmap(role: str):

    for key in ROLE_ROADMAPS.keys():

        if key.lower() == role.lower():

            return jsonify({
                "role": key,
                "steps": ROLE_ROADMAPS[key]
            })

    return jsonify({
        "error": f"Role '{role}' not found"
    }), 404


@app.route("/recommendations", methods=["POST"])
def get_recommendations():

    data = request.get_json(force=True) or {}

    interest = data.get("interest")
    level = data.get("level")

    if interest not in INTERESTS:

        return jsonify({
            "error": "Invalid or missing 'interest'",
            "allowed_interests": INTERESTS
        }), 400

    if level not in LEVELS:

        return jsonify({
            "error": "Invalid or missing 'level'",
            "allowed_levels": LEVELS
        }), 400

    recs = generate_recommendations(
        interest,
        level
    )

    return jsonify({
        "interest": interest,
        "level": level,
        "recommendations": recs
    })


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(force=True) or {}

    message = (
        data.get("message") or ""
    ).strip()

    if not message:

        return jsonify({
            "answer": "Please type a question."
        })

    answer = ask_llm(message)

    formatted = format_as_bullets(answer)

    return jsonify({
        "answer": formatted
    })


@app.route("/mentor", methods=["GET"])
def mentor_ui():
    """
    Direct access to the NEXORA mentor UI.
    """

    return render_template("index.html")


# ============================================================
# OPTIONAL GROQ DIAGNOSTIC ENDPOINT
# ============================================================

@app.route("/groq-status", methods=["GET"])
def groq_status():
    """
    Diagnostic endpoint.

    Does NOT expose the API key.
    """

    if not GROQ_API_KEY:
        return jsonify({
            "status": "error",
            "message": "GROQ_API_KEY is not configured",
            "model": GROQ_MODEL
        }), 500

    if groq_client is None:
        return jsonify({
            "status": "error",
            "message": "Groq client is not initialized",
            "model": GROQ_MODEL
        }), 500

    try:

        model = groq_client.models.retrieve(
            GROQ_MODEL
        )

        return jsonify({
            "status": "success",
            "message": "Groq model is accessible",
            "model": model.id
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e),
            "model": GROQ_MODEL
        }), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=False
    )

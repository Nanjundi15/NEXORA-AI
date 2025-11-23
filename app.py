import os
from typing import List
from flask import Flask, jsonify, request, render_template
from groq import Groq

app = Flask(__name__)

# ----------------- STATIC DATA (same as your React advisor) -----------------

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

    if interest == "Data Science":
        recs.extend([
            "Learn statistics, probability, and EDA thoroughly.",
            "Practice SQL queries on realistic datasets.",
            "Create dashboards with Power BI / Tableau.",
            "Work on case studies: churn prediction, sales forecasting, A/B testing.",
        ])

    if interest == "Web Development":
        recs.extend([
            "Finish HTML, CSS, and modern JavaScript (ES6+).",
            "Learn React and build at least 3 responsive UIs.",
            "Connect frontend to a simple REST API backend.",
            "Deploy your apps to Vercel / Netlify / Render.",
        ])

    if interest == "Cloud & DevOps":
        recs.extend([
            "Understand Linux basics and shell scripting.",
            "Pick one cloud provider (Azure recommended for you 😉).",
            "Learn Docker and basics of CI/CD.",
            "Deploy at least one end-to-end project to the cloud.",
        ])

    if interest == "Cybersecurity":
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


# ----------------- GROQ LLM CONFIG -----------------

# Set this in PowerShell before running:
#   $env:GROQ_API_KEY="gsk-xxxxxxxxxxxxxxxxxxxx"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set. LLM-based chat will not work until you set it.")

groq_client = Groq(api_key=GROQ_API_KEY)


def build_llm_prompt(user_question: str) -> str:
    """
    Prompt that tells Groq LLaMA3 to behave like NEXORA AI Advisor,
    using same idea as your React dashboard.
    """
    prompt = f"""
You are NEXORA Dynamic AI Advisor.

You help students and freshers with:
- Choosing roles: AI Engineer, Data Scientist, Full-Stack Developer, Cloud & DevOps, Cybersecurity
- Planning learning paths (Beginner, Intermediate, Advanced)
- Interview preparation, projects, and placements

Answer the user's question clearly, practically, and in 4–8 sentences.
If it is a roadmap / plan question, give step-by-step bullet points.

User question:
{user_question}
"""
    return prompt.strip()


def ask_llm(message: str) -> str:
    """Call Groq LLaMA3 chat completion and return the text answer."""
    if not GROQ_API_KEY:
        return "⚠ GROQ_API_KEY is not set on the server. Please configure it first."

    prompt = build_llm_prompt(message)

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI mentor for careers, courses, and skills.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Groq error:", e)
        return f"⚠ Groq error: {e}"


def format_as_bullets(text: str) -> str:
    """
    Convert multi-line or paragraph response into bullet-style formatting.
    Adds bullets and bolds 'Month', 'Step' and colon-ended headings.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    formatted = []

    for line in lines:
        lower = line.lower()
        if lower.startswith("month") or lower.startswith("step") or line.endswith(":"):
            formatted.append(f"\n\n**{line}**")
        else:
            formatted.append(f"• {line}")

    return "\n".join(formatted)


# ----------------- FLASK ROUTES -----------------

@app.route("/", methods=["GET"])
def home():
    """Simple root route."""
    return jsonify({
        "app": "NEXORA Course & Roadmap Advisor + Groq AI",
        "description": "Provides static course/roadmap data and an AI mentor endpoint.",
        "endpoints": {
            "/interests": "GET - list available interests/tracks",
            "/levels": "GET - list available levels",
            "/role_roadmaps": "GET - list all roles and roadmaps",
            "/role_roadmaps/<role>": "GET - roadmap for a specific role",
            "/recommendations": "POST - body: {interest, level} -> learning path",
            "/chat": "POST - body: {message} -> Groq LLaMA3 AI answer",
            "/mentor": "GET - HTML UI page",
        }
    })


@app.route("/interests", methods=["GET"])
def get_interests():
    return jsonify({"interests": INTERESTS})


@app.route("/levels", methods=["GET"])
def get_levels():
    return jsonify({"levels": LEVELS})


@app.route("/role_roadmaps", methods=["GET"])
def get_all_roadmaps():
    return jsonify({"role_roadmaps": ROLE_ROADMAPS})


@app.route("/role_roadmaps/<role>", methods=["GET"])
def get_role_roadmap(role: str):
    # case-insensitive match
    for key in ROLE_ROADMAPS.keys():
        if key.lower() == role.lower():
            return jsonify({"role": key, "steps": ROLE_ROADMAPS[key]})
    return jsonify({"error": f"Role '{role}' not found"}), 404


@app.route("/recommendations", methods=["POST"])
def get_recommendations():
    """
    Expect JSON like:
    {
        "interest": "Artificial Intelligence",
        "level": "Beginner"
    }
    """
    data = request.get_json(force=True) or {}
    interest = data.get("interest")
    level = data.get("level")

    if interest not in INTERESTS:
        return jsonify({"error": "Invalid or missing 'interest'", "allowed_interests": INTERESTS}), 400
    if level not in LEVELS:
        return jsonify({"error": "Invalid or missing 'level'", "allowed_levels": LEVELS}), 400

    recs = generate_recommendations(interest, level)
    return jsonify({
        "interest": interest,
        "level": level,
        "recommendations": recs,
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"answer": "Please type a question."})

    answer = ask_llm(message)
    formatted = format_as_bullets(answer)
    return jsonify({"answer": formatted})


@app.route("/mentor", methods=["GET"])
def mentor_ui():
    """Serve the HTML UI page."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

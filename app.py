import os
import re
import time
import joblib
import logging
import warnings

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CSV_FILE_PATH = os.getenv("CSV_FILE_PATH", "trouble_ticket_data.csv")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

app = Flask(__name__)
CORS(app)


class CSVChatGPTAssistant:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.llm = None
        self.agent = None

        self.model = None
        self.label_encoders = None
        self.features = None

        self.chat_history = []

        self.load_data()
        self.load_llm()
        self.load_ml_model()
        self.create_agent()

    def load_data(self):
        encodings = ["utf-8", "latin1", "cp1252"]

        for enc in encodings:
            try:
                self.df = pd.read_csv(
                    self.csv_path,
                    encoding=enc,
                    low_memory=False,
                    on_bad_lines="skip"
                )
                print(f"CSV loaded using encoding: {enc}")
                break
            except Exception as e:
                print(f"Failed encoding {enc}: {e}")

        if self.df is None:
            print("CSV file could not be loaded.")
            return

        self.df.columns = self.df.columns.str.strip().str.upper()
        self.df = self.df.loc[:, ~self.df.columns.str.contains("^UNNAMED")]

        for col in self.df.select_dtypes(include=["object"]).columns:
            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .replace(["nan", "None", "null", "NaN"], np.nan)
            )

        if "TT_CREATION_TIME" in self.df.columns:
            self.df["TT_CREATION_TIME"] = pd.to_datetime(
                self.df["TT_CREATION_TIME"],
                errors="coerce",
                dayfirst=True
            )

        if "SATISFACTIONSCORE" in self.df.columns:
            self.df["SATISFACTION_NUMERIC"] = (
                self.df["SATISFACTIONSCORE"]
                .astype(str)
                .str.lower()
                .str.replace("star", "", regex=False)
                .str.replace(r"[^0-9]", "", regex=True)
            )

            self.df["SATISFACTION_NUMERIC"] = pd.to_numeric(
                self.df["SATISFACTION_NUMERIC"],
                errors="coerce"
            )

            median_value = self.df["SATISFACTION_NUMERIC"].median()

            if pd.isna(median_value):
                median_value = 3

            self.df["SATISFACTION_NUMERIC"] = self.df["SATISFACTION_NUMERIC"].fillna(median_value)

        self.df.drop_duplicates(inplace=True)

        print(f"Dataset loaded: {len(self.df):,} rows and {len(self.df.columns)} columns")

    def load_llm(self):
        try:
            if not GROQ_API_KEY:
                print("GROQ_API_KEY is missing.")
                return

            self.llm = ChatGroq(
                model=MODEL_NAME,
                temperature=0,
                api_key=GROQ_API_KEY,
                timeout=120,
                max_retries=3
            )

            print(f"Groq LLM loaded: {MODEL_NAME}")

        except Exception as e:
            print(f"LLM loading failed: {e}")
            self.llm = None

    def load_ml_model(self):
        try:
            self.model = joblib.load("models/xgboost_complaint_model.pkl")
            self.label_encoders = joblib.load("models/label_encoders.pkl")
            self.features = joblib.load("models/model_features.pkl")
            print("ML prediction model loaded.")
        except Exception as e:
            print(f"ML model not loaded: {e}")
            self.model = None
            self.label_encoders = None
            self.features = None

    def create_agent(self):
        if self.df is None or self.llm is None:
            return

        try:
            self.agent = create_pandas_dataframe_agent(
                llm=self.llm,
                df=self.df,
                verbose=True,
                allow_dangerous_code=True,
                handle_parsing_errors=True,
                max_execution_time=120,
                max_iterations=15,
                agent_executor_kwargs={
                    "handle_parsing_errors": True
                }
            )
            print("CSV pandas agent created.")

        except Exception as e:
            print(f"Agent creation failed: {e}")
            self.agent = None

    def add_history(self, user_query, assistant_answer):
        self.chat_history.append(f"User: {user_query}")
        self.chat_history.append(f"Assistant: {assistant_answer}")

        if len(self.chat_history) > 12:
            self.chat_history = self.chat_history[-12:]

    def history_text(self):
        return "\n".join(self.chat_history[-10:])

    def extract_user_id(self, query):
        q = query.lower()

        patterns = [
            r"user\s*id\s+([a-zA-Z0-9_@.-]+)",
            r"user\s+([a-zA-Z0-9_@.-]+)",
            r"customer\s*id\s+([a-zA-Z0-9_@.-]+)",
            r"id\s+([a-zA-Z0-9_@.-]+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, q)
            if match:
                return match.group(1)

        return None

    def route_intent(self, query):
        q = query.lower()

        # Specific user-level prediction only
        if (
            "predict" in q
            or "prediction" in q
            or "next month" in q
            or ("chance" in q and "user" in q)
            or ("risk" in q and "user" in q)
        ):
            return "prediction"

        # CSV analysis questions
        if any(word in q for word in [
            "how many", "count", "total", "top", "most common", "least common",
            "average", "mean", "minimum", "maximum", "highest", "lowest",
            "summary", "overview", "columns", "show", "list", "which user",
            "which city", "distribution", "breakdown", "tickets", "records",
            "complaints", "fault", "subfault", "satisfaction"
        ]):
            return "csv_analysis"

        # Explanation/recommendation questions
        if any(word in q for word in [
            "explain", "why", "recommend", "suggest", "insight",
            "reason", "meaning", "interpret", "conclusion", "solution",
            "reduce", "improve", "risk"
        ]):
            return "csv_reasoning"

        # Project/code help
        if any(word in q for word in [
            "code", "flask", "api", "model", "xgboost", "machine learning",
            "ml", "project", "assistant", "chatbot", "csv agent"
        ]):
            return "project_help"

        return "csv_chat"

    def direct_csv_answer(self, query):
        q = query.lower()

        if self.df is None:
            return "Dataset is not loaded."

        if "columns" in q:
            return "Columns in your CSV:\n\n" + "\n".join(self.df.columns)

        if "summary" in q or "overview" in q:
            response = f"""
Dataset Summary

Total Records/Tickets: {len(self.df):,}
Total Columns: {len(self.df.columns)}

Columns:
{", ".join(self.df.columns)}
"""

            if "FAULTTYPE" in self.df.columns:
                response += "\n\nTop 5 Fault Types:\n"
                response += self.df["FAULTTYPE"].value_counts().head(5).to_string()

            if "SATISFACTION_NUMERIC" in self.df.columns:
                response += f"\n\nAverage Satisfaction Score: {self.df['SATISFACTION_NUMERIC'].mean():.2f}"

            return response.strip()

        user_id = self.extract_user_id(query)

        if user_id and any(w in q for w in ["how many", "count", "complaints", "tickets", "total"]):
            if "USERID" in self.df.columns:
                count = len(
                    self.df[
                        self.df["USERID"].astype(str).str.lower() == user_id.lower()
                    ]
                )
                return f"User ID `{user_id}` has made {count:,} tickets/complaints."

        if (
            "total tickets" in q
            or "total records" in q
            or "how many tickets" in q
            or "how many records" in q
            or "how many complaints" in q
        ):
            return f"There are {len(self.df):,} total tickets/records in your CSV."

        if "most common complaint" in q or "most common fault" in q:
            if "FAULTTYPE" in self.df.columns:
                counts = self.df["FAULTTYPE"].value_counts()
                return f"The most common fault type is `{counts.idxmax()}` with {counts.max():,} tickets."

        if "top" in q and "fault" in q:
            if "FAULTTYPE" in self.df.columns:
                return "Top fault types:\n\n" + self.df["FAULTTYPE"].value_counts().head(10).to_string()

        if "top" in q and "subfault" in q:
            if "SUBFAULTTYPE" in self.df.columns:
                return "Top subfault types:\n\n" + self.df["SUBFAULTTYPE"].value_counts().head(10).to_string()

        if "city" in q and ("top" in q or "most" in q or "complaints" in q):
            if "CITY" in self.df.columns:
                return "Cities with most complaints:\n\n" + self.df["CITY"].value_counts().head(10).to_string()

        if "which user" in q and ("most" in q or "highest" in q):
            if "USERID" in self.df.columns:
                return "Users with most complaints:\n\n" + self.df["USERID"].value_counts().head(10).to_string()

        if "average satisfaction" in q or "mean satisfaction" in q:
            if "SATISFACTION_NUMERIC" in self.df.columns:
                avg = self.df["SATISFACTION_NUMERIC"].mean()
                return f"The average satisfaction score is {avg:.2f} stars."

        if "lowest satisfaction" in q and "fault" in q:
            if "FAULTTYPE" in self.df.columns and "SATISFACTION_NUMERIC" in self.df.columns:
                result = (
                    self.df.groupby("FAULTTYPE")["SATISFACTION_NUMERIC"]
                    .mean()
                    .sort_values()
                    .head(10)
                )
                return "Fault types with lowest average satisfaction:\n\n" + result.to_string()

        if "highest satisfaction" in q and "fault" in q:
            if "FAULTTYPE" in self.df.columns and "SATISFACTION_NUMERIC" in self.df.columns:
                result = (
                    self.df.groupby("FAULTTYPE")["SATISFACTION_NUMERIC"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                )
                return "Fault types with highest average satisfaction:\n\n" + result.to_string()

        if "satisfaction" in q and "distribution" in q:
            if "SATISFACTION_NUMERIC" in self.df.columns:
                return "Satisfaction score distribution:\n\n" + (
                    self.df["SATISFACTION_NUMERIC"]
                    .value_counts()
                    .sort_index()
                    .to_string()
                )

        return None

    def clean_agent_error_output(self, error_text):
        """
        If LangChain throws an OUTPUT_PARSING_FAILURE but the LLM already produced
        a useful answer, this function extracts that answer and hides the ugly error.
        """

        patterns = [
            r"Could not parse LLM output:\s*`(.*?)`",
            r"Could not parse LLM output:\s*(.*?)(?:For troubleshooting|$)"
        ]

        for pattern in patterns:
            match = re.search(pattern, error_text, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
                cleaned = cleaned.replace("\\n", "\n")
                cleaned = cleaned.strip("`")
                return cleaned

        return None

    def predict_user(self, user_id):
        try:
            if self.model is None:
                return "ML model is not loaded. Please run `train_model.py` first."

            if "USERID" not in self.df.columns:
                return "USERID column was not found in the CSV."

            user_data = self.df[
                self.df["USERID"].astype(str).str.lower() == user_id.lower()
            ]

            if user_data.empty:
                return f"No data found for user ID `{user_id}`."

            if "TT_CREATION_TIME" in user_data.columns:
                user_data = user_data.sort_values("TT_CREATION_TIME")

            latest = user_data.iloc[-1]
            input_data = {}

            for col in self.features:
                if col == "SATISFACTION_NUMERIC":
                    val = latest.get(col, 3)
                    input_data[col] = 3 if pd.isna(val) else float(val)
                else:
                    val = str(latest.get(col, "Unknown"))

                    if self.label_encoders and col in self.label_encoders:
                        le = self.label_encoders[col]

                        if val not in le.classes_:
                            val = le.classes_[0]

                        input_data[col] = le.transform([val])[0]
                    else:
                        input_data[col] = 0

            X_input = pd.DataFrame([input_data])
            X_input = X_input[self.features]

            probability = self.model.predict_proba(X_input)[0][1]
            percent = round(probability * 100, 2)

            if percent >= 70:
                risk = "High"
            elif percent >= 40:
                risk = "Medium"
            else:
                risk = "Low"

            return f"""
Prediction Result

User ID: {user_id}
Complaint Chance Next Month: {percent}%
Risk Level: {risk}

Latest Record Used:
Fault Type: {latest.get("FAULTTYPE", "N/A")}
Sub Fault Type: {latest.get("SUBFAULTTYPE", "N/A")}
Satisfaction Score: {latest.get("SATISFACTIONSCORE", "N/A")}

Explanation:
The model used this user's latest ticket information to estimate the chance of another complaint next month.
""".strip()

        except Exception as e:
            return f"Prediction error: {str(e)}"

    def llm_csv_response(self, query, intent):
        if self.llm is None:
            return "LLM is not available. Please check your GROQ_API_KEY."

        if self.agent is None:
            return self.llm_general_response(query, intent)

        prompt = f"""
You are a CSV-focused ChatGPT-style AI assistant.

You provide ChatGPT-like services, but only for CSV data and this project.

You can:
- Chat naturally
- Answer CSV questions
- Analyze dataframe df
- Summarize records
- Explain results
- Give recommendations from the CSV
- Help with Flask, ML model, XGBoost, prediction, and project explanation
- Answer supervisor-style questions

Very important rules:
1. For dataset questions, use pandas on dataframe df.
2. Never invent numbers.
3. Never guess CSV values.
4. If exact data is available, calculate it.
5. If a column is missing, say clearly that the column is not present.
6. Keep final answer clear and student-friendly.
7. Do not show Python code unless the user asks for code.
8. Do not mention internal tool errors.
9. Always provide a final natural-language answer.
10. If question is unrelated to CSV, trouble tickets, ML prediction, Flask, or this project, politely redirect.

Conversation history:
{self.history_text()}

Dataset information:
Rows: {len(self.df)}
Columns: {list(self.df.columns)}

Current user intent:
{intent}

User question:
{query}

Answer clearly.
"""

        try:
            result = self.agent.invoke({"input": prompt})

            if isinstance(result, dict):
                return result.get("output", str(result))

            return str(result)

        except Exception as e:
            error_text = str(e)
            cleaned_answer = self.clean_agent_error_output(error_text)

            if cleaned_answer:
                return cleaned_answer

            return self.llm_general_response(query, intent)

    def llm_general_response(self, query, intent):
        if self.llm is None:
            return "LLM is not available. Please check your GROQ_API_KEY."

        prompt = f"""
You are a CSV-focused ChatGPT-style assistant.

Your main domain:
- CSV analysis
- Trouble ticket dataset
- Fault types
- Subfault types
- Satisfaction score
- User complaint history
- ML prediction
- Flask chatbot project

Conversation history:
{self.history_text()}

Dataset:
Rows: {len(self.df) if self.df is not None else 0}
Columns: {list(self.df.columns) if self.df is not None else []}

User intent:
{intent}

User question:
{query}

Answer like ChatGPT, but keep the answer focused on CSV analysis or this assistant project.
If the question is unrelated, politely redirect to CSV/trouble-ticket analysis.
"""

        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            return f"LLM error: {str(e)}"

    def ask(self, query):
        start = time.time()

        if self.df is None:
            return {
                "success": False,
                "response": "Dataset is not loaded.",
                "processing_time": 0,
                "source": "error"
            }

        intent = self.route_intent(query)

        try:
            if intent == "prediction":
                user_id = self.extract_user_id(query)

                if not user_id:
                    response = "Please provide a user ID. Example: `Predict complaint chance for user id maibad`"
                    source = "router"
                else:
                    response = self.predict_user(user_id)
                    source = "xgboost_prediction"

            elif intent == "csv_analysis":
                response = self.direct_csv_answer(query)

                if response is None:
                    response = self.llm_csv_response(query, intent)
                    source = "csv_agent"
                else:
                    source = "direct_pandas"

            else:
                response = self.llm_csv_response(query, intent)
                source = "csv_focused_llm"

            self.add_history(query, response)

            return {
                "success": True,
                "response": response,
                "processing_time": round(time.time() - start, 2),
                "source": source,
                "intent": intent
            }

        except Exception as e:
            fallback = self.direct_csv_answer(query)

            if fallback:
                self.add_history(query, fallback)

                return {
                    "success": True,
                    "response": fallback,
                    "processing_time": round(time.time() - start, 2),
                    "source": "fallback_pandas",
                    "intent": intent
                }

            response = f"Assistant error: {str(e)}"
            self.add_history(query, response)

            return {
                "success": False,
                "response": response,
                "processing_time": round(time.time() - start, 2),
                "source": "error",
                "intent": intent
            }

    def overview(self):
        if self.df is None:
            return {"error": "Dataset not loaded"}

        return {
            "assistant_type": "CSV-focused ChatGPT-style assistant",
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "column_names": list(self.df.columns),
            "model_loaded": self.model is not None,
            "llm_loaded": self.llm is not None,
            "agent_loaded": self.agent is not None,
            "groq_model": MODEL_NAME
        }


csv_agent = CSVChatGPTAssistant(CSV_FILE_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "No query provided"}), 400

    return jsonify(csv_agent.ask(query))


@app.route("/api/dataset_info", methods=["GET"])
def dataset_info():
    return jsonify(csv_agent.overview())


@app.route("/api/preview", methods=["GET"])
def preview():
    if csv_agent.df is None:
        return jsonify({"error": "Dataset not loaded"}), 500

    return jsonify(csv_agent.df.head(10).to_dict(orient="records"))


@app.route("/api/reset_chat", methods=["POST"])
def reset_chat():
    csv_agent.chat_history = []
    return jsonify({"success": True, "message": "Chat history cleared."})

    
if __name__ == "__main__":
    print("\nCSV-Focused ChatGPT-Style Assistant Started")
    print("------------------------------------------")
    print(csv_agent.overview())

    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
    
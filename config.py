import os
from pathlib import Path
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

# Paths
RESOURCES_PATH = r"/home/zamlamb/KdG/agenticrag/Resources"
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# LangSmith Configuration (loaded from .env)
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "workstations")
client = Client()

# Collection names for vector stores
WORK_COLLECTION = "work_instructions_RAG"  # For work instructions (Qdrant)
SOP_COLLECTION = "sop"                     # For SOPs (Chroma)

# Benchmark Reference for style guidance
BENCHMARK_REFERENCE = """
Benchmark Reference Questions and Answers:
1. Which personal protective equipment do I need to use during Hypophosphorous Acid Addition?
   Answer: Neoprene gloves, face shield for the helmet and an arpon.
2. Describe in detail the steps that I need to do for the Sulfated Analysis by HPLC.
   Answer:
      1. Puncture the sample. Open the slot to insert the needle and introduce the sample. Remove the sample and close the slot.
      2. Enter the sample name. In Method, select the corresponding base for the analysis to be performed. Press Inject.
      3. Click on Integrate and Quantitate. Click on Send data to review to visualize the results.
      4. Click on the icon, and the window "Specify single inject Parameters" will appear.
3. Provide me the raw materials to be used for the synthesis of Alkylbenzen Sulfonic Acid and their SAP Code.
   Answer:
      • Linear Alkylbenzene (50386977)
      • Liquid Sulfur (50087496)
      • Sodium Hydroxide 30% (50059005)
      • Water (50197449)
4. When synthesizing Alkylbenzen sulfonic acid, which should be the setpoint for the sulfur trioxide when doing the sulfonation?
   Answer: 6%
5. Which range of humidity values are acceptable for the Alkylbenzen Sulfonic Acid?
   Answer: Between 0.5 and 2%.
6. Describe the hazard classification of the AN-84 product.
   Answer:
      • H302: Harmful if swallowed
      • H314: Causes severe skin burns and eye damage.
      • H400: Very toxic to aquatic organisms.
7. Describe the amidation reaction for the production of AN-84.
   Answer:
      1. Maintain 170 ºC for 3 hours. Monitor that pressure does not exceed 2500 mBar.
      2. Take a sample and analyze the Ester Band; if above threshold, refine conditions.
8. Describe, in detail, the operational method for the production of Texapon S80.
   Answer:
      1. Check that there is enough raw material.
      2. Follow the step-by-step process as per the operational guidelines.
"""

# (Other configuration items like Azure connection string, etc.)

# Azure project connection string
AZURE_PROJECT_CONN_STRING = os.getenv("AZURE_PROJECT_CONN_STRING")

# Rwanda Energy Consumption Predictor API



## Setup Instructions

1. Activate Python Environment
   Make sure you have your Python virtual environment ready. To activate it, run:  
   bash
   .\env-supervised\Scripts\Activate


2. Install Dependencies
   If you already have a `requirements.txt` file, install all dependencies using:

   bash
   pip install -r requirements.txt
   

3. Run the API
   Start the FastAPI server with the following command:

   bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000


4. Swagger UI
   Once the server is running, you can access the interactive API documentation (Swagger UI) at:
  (http://127.0.0.1:8000/docs)

5. Freeze Dependencies
   To save your current environment packages, run:

   bash
   pip freeze > requirements.txt
   

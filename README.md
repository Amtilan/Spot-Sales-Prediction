# Restaurant Order Prediction API

A FastAPI-based service for predicting restaurant orders based on menu items, order types, and specific areas.

## Endpoint(s)
### 1. `/predict` [POST]
- **Description**: Get predictions for a specific area and date.
- **Request**:
  ```json
  {
      "area": "AreaName",
      "target_date": "YYYY-MM-DD"
  }
  ```
- **Response**:
  ```json
  {
      "predictions": [
          {
              "Category": "CategoryName",
              "Item": "ItemName",
              "Type_of_Order": "Takeaway/Dining",
              "Predicted_Orders": 50
          }
      ],
      "total_orders": 100,
      "area": "AreaName",
      "target_date": "YYYY-MM-DD"
  }
  ```

### 2. `/docs` [GET]
- **Description**: Access the Swagger to explore available endpoints.


---

## Running the Application

### Option 1: Local Installation
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Start the server:
    ```bash
    python -m uvicorn main:app --reload
    ```
3. Access the API at `http://127.0.0.1:8000`

### Option 2: Using Docker
1. Build the image:
    ```bash
    docker build -t fastapi-app .
    ```
2. Run the container:
    ```bash
    docker run -p 8000:8000 fastapi-app
    ```
3. Access the API at `http://127.0.0.1:8000`

---
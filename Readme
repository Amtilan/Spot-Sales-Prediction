# Restaurant Order Prediction API

A FastAPI-based service for predicting restaurant orders based on menu items, order types, and specific areas.

## Endpoints

### 1. `/last_date` [GET]
- **Description**: Retrieve the most recent date in the dataset.
- **Response**:
  ```json
  {
      "last_date": "YYYY-MM-DD"
  }
  ```

### 2. `/reload-data` [POST]
- **Description**: Manually reload the data from the CSV file. Usually you don't need that, because it will automatically reload the data from table once a day.

### 3. `/menu-items` [GET]
- **Description**: Fetch all menu items grouped by category.
- **Response**:
  ```json
  {
      "categories": {
          "Category1": ["Item1", "Item2"],
          "Category2": ["Item3", "Item4"]
      },
      "total_items": 10
  }
  ```

### 4. `/predict` [POST]
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

### 5. `/docs` [GET]
- **Description**: Access the Swagger to explore available endpoints.


---

## Running the Application

### Option 1: Local Installation
1. Install dependencies:
    ```bash
    pip install fastapi uvicorn pandas numpy scikit-learn apscheduler
    ```
2. Start the server:
    ```bash
    uvicorn main:app --reload
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
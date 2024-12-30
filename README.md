
# 🚀 **MLOps Pipeline for Diabetes Prediction**

![MLOps Pipeline](https://sl.bing.net/iSIjpEM1k0i)

Welcome to the **Diabetes Prediction MLOps Pipeline**! This repository showcases the seamless integration of MLOps tools and techniques to build, train, deploy, and monitor a machine learning model for predicting diabetes. 🌟

---

## 🗂️ **Project Directory Structure**

Here's an overview of the project's structure:

| **Folder/File**          | **Description**                                                                 |
|---------------------------|---------------------------------------------------------------------------------|
| 📁 `data/`               | Contains raw and processed datasets.                                           |
| 📁 `data_schema/`        | Schema files for data validation.                                              |
| 📁 `diabetes/`           | Contains the core project directories for FastAPI and related components. See below:|
| ├── 📁 `cloud/`          | Handles cloud-related utilities for the project.                              |
| ├── 📁 `components/`     | Contains modular components for the FastAPI project.                          |
| ├── 📁 `constant/`       | Stores constant variables used across the project.                            |
| ├── 📁 `entity/`         | Defines data entities and schemas.                                            |
| ├── 📁 `exception/`      | Custom exception handling modules.                                            |
| ├── 📁 `logging/`        | Logging configurations for better debugging.                                  |
| ├── 📁 `pipeline/`       | Defines the pipelines for ingestion, training, and deployment.                |
| ├── 📁 `utils/`          | Utility scripts for supporting functions.                                     |
| └── 📄 `__init__.py`     | Marks the directory as a Python package.                                      |
| 📁 `final_model/`        | Final trained models and artifacts.                                            |
| 📁 `notebook/`           | Jupyter notebooks for EDA and model experimentation.                          |
| 📁 `prediction_output/`  | Stores output predictions.                                                     |
| 📁 `templates/`          | HTML templates for the web interface (if applicable).                         |
| 📄 `.gitignore`          | Lists files and folders to be ignored by Git.                                  |
| 📄 `LICENSE`             | License information for this project.                                          |
| 📄 `README.md`           | The file you are reading right now!                                            |
| 📄 `app.py`              | FastAPI application for serving the model.                                     |
| 📄 `main.py`             | Script for training the model.                                                 |
| 📄 `pushdata.py`         | Script for ingesting data into the system.                                     |
| 📄 `requirements.txt`    | Lists all dependencies required for this project.                              |
| 📄 `setup.py`            | Configuration file for setting up the project.                                 |
| 📄 `test.csv`            | Test dataset for predictions.                                                  |
| 📄 `testDB.py`           | Script for testing database connections.                                       |

---

## 🔧 **Features**

✨ **Data Ingestion**
   - Handles raw data ingestion from multiple sources and validates it using schema.

✨ **Model Training**
   - Train and evaluate the machine learning model using robust tools.

✨ **API Integration**
   - Serve the trained model as an API endpoint using FastAPI. 🖥️

✨ **Experiment Tracking**
   - Log metrics, hyperparameters, and artifacts with MLFlow.

✨ **CI/CD Pipelines**
   - Automate build, test, and deployment processes with GitHub Actions. 🤖

✨ **Containerization**
   - Dockerize the application and manage it via AWS Elastic Container Registry (ECR).

✨ **Cloud Storage**
   - Store datasets, artifacts, and models in AWS S3 bucket.

✨ **Database**
   - MongoDB for storing structured input data.

✨ **Deployment**
   - Deployed model on AWS EC2 instance for scalable serving. 🌐

---

## 🎯 **How to Get Started**

### **Prerequisites:**
- Python 3.8+
- Docker
- AWS CLI
- MongoDB instance

### **Steps:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SivakumarReddy143/diabetes-prediction-mlops.git
   cd diabetes-prediction-mlops
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Create a `.env` file and specify credentials for MongoDB, AWS, and MLFlow.

4. **Run the FastAPI app:**
   ```bash
   uvicorn app:app --reload
   ```

5. **Access the API:**
   - Open your browser and navigate to `http://127.0.0.1:8000/docs` to explore the API.

6. **Run tests:**
   ```bash
   python testDB.py
   ```

---

## 🎥 **Demo**

![Demo](https://media.giphy.com/media/l41lFw057lAJQMwg0/giphy.gif)

---

## 🛠️ **Future Enhancements**

- Implement monitoring with Prometheus and Grafana.
- Add multi-cloud support for deployment.
- Develop a user-friendly front-end interface. 🌈

---

## 🤝 **Contributing**

We welcome contributions from the community! 💡 To contribute:
1. Fork this repository.
2. Create a new branch.
3. Make your changes and submit a pull request.

---

## 📝 **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more information.

---

## 📞 **Contact**

For any questions, feel free to reach out to:
- **Sivakumar Reddy**
- 📧 Email: [mshivakumarreddy78@gmail.com](mailto:mshivakumarreddy78@gmail.com)
- 🌐 GitHub: [SivakumarReddy143](https://github.com/SivakumarReddy143)

![Thank You](https://media.giphy.com/media/26u4nJPf0JtQPdStq/giphy.gif)
```


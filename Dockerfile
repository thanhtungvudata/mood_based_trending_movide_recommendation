# Use official Python image
FROM python:3.11-slim

# Set the working directory inside Docker
WORKDIR /app

# Copy everything into Docker (including app.py)
COPY . /app
COPY models /app/models

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# # Expose Streamlit's default port
# EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

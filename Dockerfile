# Use the official Python image as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy the project files to the container
COPY . /app

# Install dependencies using Poetry
# Using --no-root to avoid installing the project itself, as we are using Poetry only for dependency management
RUN poetry config virtualenvs.create false \
    && poetry install --no-root

# Expose the port Streamlit will run on
EXPOSE 8501

# Add an environment variable to specify the mode
ENV APP_MODE=app

# Default command to run the application in app mode
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# Copy the requirements.txt file to the container
COPY ../requirements.txt ../requirements-dev.txt ./

# Install the requirements
RUN pip install -r requirements-dev.txt

FROM continuumio/miniconda3
WORKDIR /app
COPY environment.yml .
RUN conda env createZ -f environment.yml -y

RUN echo "source activate mlp-scratch" > ~/.bashrc \
    && /bin/bash -c "source ~/.bashrc"

# Copy your application code into the container
COPY . .

# Set the default command to run when the container starts
# This ensures your application runs within the activated Conda environment
CMD ["conda", "run", "-n", "mlp-scratch", "gunicorn", "src.wsgi:app"]

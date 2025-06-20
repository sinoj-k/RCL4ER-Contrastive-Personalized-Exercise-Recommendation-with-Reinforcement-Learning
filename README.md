# RCL4ER-Contrastive-Personalized-Exercise-Recommendation-with-Reinforcement-Learning
The goal of this project is to recommend personalized aptitude exercises to students by leveraging AI techniquesFeatures
- Deep Knowledge Tracing (DKT): Models students' evolving knowledge states.
- Data Augmentation: Enhances interaction sequences with masking, permutation, and replacement.
- Contrastive Learning: Improves sequence embeddings through self-supervised training.
- Reinforcement Learning: Selects the best-fit questions for personalized learning.
- Top-K Recommendations: Suggests the most relevant questions per student.
Technologies Used
- Python, PyTorch
- NumPy, Pandas, Scikit-learn
- Jupyter Notebook
1. Clone the Repository:
git clone https://github.com/yourusername/rcl4er.git
cd rcl4er
2. Install Dependencies:
pip install -r requirements.txt
Model Training
Train the full RCL4ER system:
python train.py
Evaluate model performance:
python evaluate.py
Outputs
Example Output:
Student ID: 3
Recommended Exercises: [Skill_8, Skill_2, Skill_5]
Evaluation Metrics:
- Accuracy
- Contrastive loss trend
- RL reward improvement
Key Modules
1. Deep Knowledge Tracing (DKT): Predicts knowledge state from skill sequences.
2. Data Augmentation: Uses masking, permutation, replacement.
3. Contrastive Learning: Uses InfoNCE loss to enhance embedding.
4. Reinforcement Learning: Q-learning agent for adaptive recommendation.
Contact
Author: IEEE
GitHub: https://github.com/sinoj-k
Email: sellasinoj@gmail.com
License
This project is open-source for educational purposes.

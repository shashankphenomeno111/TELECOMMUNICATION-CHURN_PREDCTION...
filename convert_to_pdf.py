from fpdf import FPDF, XPos, YPos

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(26, 95, 122)
        self.cell(0, 10, 'Telecom Churn Prediction - Interview Guide', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(26, 95, 122)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(26, 95, 122)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)
        
    def section_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(46, 134, 171)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)
        
    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(51, 51, 51)
        # Clean text of special characters
        text = text.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(190, 6, text)
        self.ln(2)
        
    def bullet_point(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(51, 51, 51)
        text = text.encode('latin-1', 'replace').decode('latin-1')
        self.cell(5, 6, '-')
        self.multi_cell(185, 6, text)
        
    def table_row(self, data, widths, is_header=False):
        if is_header:
            self.set_font('Helvetica', 'B', 9)
            self.set_fill_color(26, 95, 122)
            self.set_text_color(255, 255, 255)
        else:
            self.set_font('Helvetica', '', 9)
            self.set_fill_color(249, 249, 249)
            self.set_text_color(51, 51, 51)
        
        for i, cell in enumerate(data):
            cell_text = str(cell)[:40].encode('latin-1', 'replace').decode('latin-1')
            self.cell(widths[i], 7, cell_text, border=1, fill=True)
        self.ln()

# Create PDF
pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Title Page
pdf.add_page()
pdf.set_font('Helvetica', 'B', 24)
pdf.set_text_color(26, 95, 122)
pdf.ln(40)
pdf.cell(0, 15, 'TELECOM CUSTOMER', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 15, 'CHURN PREDICTION', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(10)
pdf.set_font('Helvetica', '', 16)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 10, 'Complete Interview Preparation Guide', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(20)
pdf.set_font('Helvetica', 'B', 12)
pdf.set_text_color(46, 134, 171)
pdf.cell(0, 8, 'Project Highlights:', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', 11)
pdf.set_text_color(51, 51, 51)
pdf.cell(0, 7, 'Dataset: 3,333 customers | 19 features', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 7, 'Best Model: XGBoost Classifier', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 7, 'Accuracy: 98% | Recall: 87%', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 7, 'Deployment: Streamlit Web App', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# Section 1: Project Overview
pdf.add_page()
pdf.chapter_title('1. PROJECT OVERVIEW')
pdf.body_text('A machine learning classification project that predicts which telecom customers are likely to churn (cancel service), enabling proactive retention strategies.')
pdf.ln(5)
pdf.section_title('Complete Workflow')
pdf.body_text('Data Loading -> EDA -> Preprocessing (StandardScaler + SMOTE) -> Feature Selection (Top 10) -> Train-Test Split (80-20) -> Model Building (RFC & XGBoost) -> Evaluation -> Deployment (Streamlit)')

# Section 2: Business Problem
pdf.add_page()
pdf.chapter_title('2. BUSINESS PROBLEM')
pdf.section_title('What is Customer Churn?')
pdf.body_text('Customer churn refers to when a customer stops using a companys service (cancels their subscription). It is also called customer attrition.')
pdf.ln(3)
pdf.section_title('Why is Churn Prediction Important?')
pdf.bullet_point('Churn Rate in Telecom: Typically 10-25% annually')
pdf.bullet_point('Revenue Loss: Each churned customer = $60-80/month lost')
pdf.bullet_point('Acquisition Cost: Getting a NEW customer costs 5x more than retaining existing one')
pdf.ln(3)
pdf.section_title('Project Objective')
pdf.body_text('Build a predictive model that identifies at-risk customers BEFORE they leave, enabling targeted retention campaigns.')

# Section 3: Dataset
pdf.add_page()
pdf.chapter_title('3. DATASET DESCRIPTION')
pdf.body_text('Total Rows: 3,333 customers | Total Columns: 19 features + 1 target | No missing values')
pdf.ln(3)
pdf.section_title('Key Features')
pdf.table_row(['Feature', 'Type', 'Description'], [50, 30, 110], is_header=True)
pdf.table_row(['total_charge', 'Numerical', 'Total monthly bill ($)'], [50, 30, 110])
pdf.table_row(['customer_service_calls', 'Numerical', 'Number of calls to support'], [50, 30, 110])
pdf.table_row(['international_plan', 'Binary', 'Has international plan (0/1)'], [50, 30, 110])
pdf.table_row(['day_mins', 'Numerical', 'Daytime call minutes'], [50, 30, 110])
pdf.table_row(['voice_mail_plan', 'Binary', 'Has voicemail plan (0/1)'], [50, 30, 110])
pdf.table_row(['churn', 'Target', '0 = Stayed, 1 = Churned'], [50, 30, 110])

# Section 4: EDA
pdf.add_page()
pdf.chapter_title('4. EXPLORATORY DATA ANALYSIS (EDA)')
pdf.section_title('4.1 Target Variable Analysis')
pdf.body_text('Class 0 (Stayed): 2,850 customers (85.5%) | Class 1 (Churned): 483 customers (14.5%)')
pdf.body_text('KEY INSIGHT: Class imbalance detected! Requires SMOTE for handling.')
pdf.ln(3)
pdf.section_title('4.2 Top Churn Predictors')
pdf.table_row(['Predictor', 'Impact', 'Churn Rate'], [60, 70, 60], is_header=True)
pdf.table_row(['Service Calls >= 4', 'HIGHEST RISK', '45%+'], [60, 70, 60])
pdf.table_row(['International Plan = Yes', '3x higher', '28% vs 11%'], [60, 70, 60])
pdf.table_row(['Total Charge > $65', 'Bill shock', 'Higher'], [60, 70, 60])
pdf.table_row(['No Voice Mail Plan', '50% more', '16% vs 8%'], [60, 70, 60])
pdf.ln(3)
pdf.section_title('4.3 Key Correlation Findings')
pdf.bullet_point('international_plan with churn: +0.26 (positive = increases churn)')
pdf.bullet_point('customer_service_calls with churn: +0.21')
pdf.bullet_point('voice_mail_plan with churn: -0.10 (negative = decreases churn)')

# Section 5: Preprocessing
pdf.add_page()
pdf.chapter_title('5. DATA PREPROCESSING')
pdf.section_title('5.1 Standard Scaling')
pdf.body_text('StandardScaler transforms features to have mean=0 and std=1.')
pdf.body_text('Formula: z = (x - mean) / std')
pdf.body_text('Why: Features have different scales (day_mins: 0-400 vs international_charge: 0-5)')
pdf.ln(3)
pdf.section_title('5.2 SMOTE for Class Imbalance')
pdf.body_text('SMOTE = Synthetic Minority Oversampling Technique')
pdf.body_text('Before SMOTE: Class 0 = 2,850 | Class 1 = 483')
pdf.body_text('After SMOTE: Class 0 = 2,850 | Class 1 = 2,850 (Balanced!)')
pdf.body_text('How: Creates synthetic minority samples by interpolating between existing samples')

# Section 6: Feature Selection
pdf.add_page()
pdf.chapter_title('6. FEATURE SELECTION')
pdf.section_title('Random Forest Feature Importance')
pdf.body_text('Used Random Forest to rank features by importance (Gini impurity reduction)')
pdf.ln(3)
pdf.table_row(['Rank', 'Feature', 'Importance'], [30, 80, 80], is_header=True)
pdf.table_row(['1', 'total_charge', '21.3%'], [30, 80, 80])
pdf.table_row(['2', 'customer_service_calls', '12.5%'], [30, 80, 80])
pdf.table_row(['3', 'day_mins', '8.9%'], [30, 80, 80])
pdf.table_row(['4', 'day_charge', '8.8%'], [30, 80, 80])
pdf.table_row(['5', 'international_plan', '7.9%'], [30, 80, 80])

# Section 7: Model Building
pdf.add_page()
pdf.chapter_title('7. MODEL BUILDING')
pdf.section_title('7.1 Train-Test Split')
pdf.body_text('80% Training (2,666 samples) | 20% Testing (667 samples)')
pdf.ln(3)
pdf.section_title('7.2 Model Comparison')
pdf.table_row(['Metric', 'Random Forest', 'XGBoost', 'Winner'], [45, 45, 45, 55], is_header=True)
pdf.table_row(['Accuracy', '97%', '98%', 'XGBoost'], [45, 45, 45, 55])
pdf.table_row(['Recall (Churners)', '81%', '87%', 'XGBoost'], [45, 45, 45, 55])
pdf.table_row(['F1-Score', '0.89', '0.93', 'XGBoost'], [45, 45, 45, 55])
pdf.table_row(['False Negatives', '20', '13', 'XGBoost'], [45, 45, 45, 55])
pdf.ln(3)
pdf.section_title('7.3 Why XGBoost?')
pdf.bullet_point('Gradient boosting - sequential error correction')
pdf.bullet_point('Built-in regularization (L1/L2) prevents overfitting')
pdf.bullet_point('Handles missing values natively')
pdf.bullet_point('Provides feature importance for interpretability')

# Section 8: Evaluation
pdf.add_page()
pdf.chapter_title('8. MODEL EVALUATION')
pdf.section_title('XGBoost Confusion Matrix')
pdf.body_text('True Negatives: 566 | False Positives: 0')
pdf.body_text('False Negatives: 13 | True Positives: 88')
pdf.ln(3)
pdf.section_title('Classification Metrics')
pdf.table_row(['Metric', 'Value', 'Meaning'], [50, 40, 100], is_header=True)
pdf.table_row(['Accuracy', '98%', 'Overall correct predictions'], [50, 40, 100])
pdf.table_row(['Precision', '100%', 'When we predict churn, we are right'], [50, 40, 100])
pdf.table_row(['Recall', '87%', 'We catch 87% of actual churners'], [50, 40, 100])
pdf.table_row(['F1-Score', '0.93', 'Balanced metric'], [50, 40, 100])
pdf.ln(3)
pdf.section_title('Why Recall Matters More')
pdf.body_text('False Negative (missing a churner) = $780/year lost revenue')
pdf.body_text('False Positive (false alarm) = Small cost (unnecessary discount)')

# Section 9: Conclusions
pdf.add_page()
pdf.chapter_title('9. CONCLUSIONS & RECOMMENDATIONS')
pdf.section_title('Key Findings')
pdf.bullet_point('14.5% churn rate - typical for telecom')
pdf.bullet_point('Service Calls >= 4 = 45%+ churn risk')
pdf.bullet_point('International plan = 3x higher churn')
pdf.bullet_point('Voicemail plan reduces churn by 50%')
pdf.ln(3)
pdf.section_title('Business Recommendations')
pdf.bullet_point('Flag customers after 2nd service call for monitoring')
pdf.bullet_point('Assign dedicated support after 3rd call')
pdf.bullet_point('Offer retention discount (15-20%) after 4th call')
pdf.bullet_point('Offer free voicemail trial to at-risk customers')
pdf.ln(3)
pdf.section_title('Business Impact')
pdf.body_text('With 100,000 customers, this model can save $5+ million annually through proactive retention.')

# Section 10: Interview Q&A Part 1 (General & EDA)
pdf.add_page()
pdf.chapter_title('10. INTERVIEW QUESTIONS & ANSWERS (PART 1)')

qa_batch_1 = [
    ('Q1: Tell me about your project', 'I developed a Customer Churn Prediction system for telecom. Using 3,333 customers with 19 features, I built an XGBoost classifier achieving 98% accuracy and 87% recall.'),
    ('Q2: What was the business problem?', 'Telecom companies face 10-25% annual churn. Each lost customer costs $60-80/month. My model identifies at-risk customers early, saving millions in revenue.'),
    ('Q3: What was your methodology?', 'EDA -> Preprocessing (Scaling + SMOTE) -> Feature Selection (Random Forest) -> Model Building (RFC vs XGBoost) -> Evaluation -> Streamlit Deployment.'),
    ('Q4: What patterns did you find in EDA?', '1) Service calls 4+ = 45%+ churn, 2) International plan = 3x churn, 3) Voicemail plan reduces churn by 50%.'),
    ('Q5: How did you handle outliers?', "Kept them! They represent real customers (heavy users/dissatisfied) who are exactly the churners we want to predict. XGBoost handles trees robustly."),
    ('Q6: What did correlation analysis show?', 'High correlation (0.99) between minutes and charges. Service calls and international plan showed positive correlation with churn.')
]

for q, a in qa_batch_1:
    pdf.section_title(q)
    pdf.body_text(a)

# Section 11: Interview Q&A Part 2 (Technical & Evaluation)
pdf.add_page()
pdf.chapter_title('11. INTERVIEW Q&A (PART 2)')

qa_batch_2 = [
    ('Q7: Why StandardScaler?', 'Normalizes features to Mean=0, Std=1. Prevents features with large ranges (day_mins) from dominating those with small ranges (charge).'),
    ('Q8: How does SMOTE work?', 'Creates synthetic minority samples by interpolating between existing ones (k-nearest neighbors) rather than just duplicating records.'),
    ('Q9: Random Forest vs XGBoost?', 'RF builds trees in parallel (bagging). XGBoost builds them sequentially (boosting) to correct previous errors, yielding 98% vs 97% accuracy.'),
    ('Q10: Explain Precision vs Recall', 'Precision (100%): No false alarms. Recall (87%): Caught 87% of actual churners. For churn, Recall is more critical to avoid losing revenue.'),
    ('Q11: Why is 98% accuracy misleading?', 'In imbalanced data (85% stay), a dummy model predicting "no churn" gets 85% accuracy but fails the business objective. We use Recall and F1-Score.'),
    ('Q12: Business impact calculation?', 'With 100K users and 15% churn, model identifies 13,050. With 50% retention success, saves $5.09M annually at $65/month revenue.')
]

for q, a in qa_batch_2:
    pdf.section_title(q)
    pdf.body_text(a)

# Section 11 Continued (Part 3)
pdf.add_page()
pdf.chapter_title('11. INTERVIEW Q&A (PART 3)')

qa_batch_3 = [
    ('Q13: How does Gradient Boosting work?', 'Builds trees sequentially where each tree predicts the residuals (errors) of the previous trees, progressively reducing the error.'),
    ('Q14: What hyperparameters would you tune?', 'n_estimators, max_depth, learning_rate, and scale_pos_weight (for imbalance) using GridSearchCV or RandomizedSearchCV.'),
    ('Q15: Explain the Confusion Matrix', 'TN: 566 (Correct stays), TP: 88 (Correct churns), FP: 0 (No false alarms), FN: 13 (Missed churners).'),
    ('Q18: What is F1-Score?', 'Harmonic mean of precision and recall. It penalizes extreme values and provides a balanced measure for imbalanced datasets.'),
    ('Q19: Explain SMOTE technically', 'Identifies k-nearest neighbors for minority samples and creates new points along the lines connecting them to increase diversity.'),
    ('Q21: How would you deploy in production?', 'Containerize with Docker, deploy on cloud (AWS/GCP), implement model monitoring for drift, and use A/B testing.')
]

for q, a in qa_batch_3:
    pdf.section_title(q)
    pdf.body_text(a)

# Section 11 Continued (Part 4)
pdf.add_page()
pdf.chapter_title('11. INTERVIEW Q&A (PART 4)')

qa_batch_4 = [
    ('Q23: Business recommendations?', '1) Intervene at 3rd service call, 2) Review intl plan pricing, 3) Use voicemail as a "sticky" feature to reduce churn.'),
    ('Q24: What would you do differently?', 'Use k-fold cross-validation, more automated hyperparameter tuning, and add model interpretability tools like SHAP.'),
    ('Q26: Cost of False Negatives?', 'Missing a churner costs ~$780/year in lost revenue. This is why we prioritize Recall over Precision.'),
    ('Q27: How to monitor drift?', 'Track prediction distribution and feature distributions over time. Retrain model if performance drops significantly.'),
    ('Q30: Why not Deep Learning?', 'For tabular data of this size, Gradient Boosting (XGBoost) typically outperforms Deep Learning and is much more interpretable.')
]

for q, a in qa_batch_4:
    pdf.section_title(q)
    pdf.body_text(a)

# Section 12: ML Fundamentals
pdf.add_page()
pdf.chapter_title('12. MACHINE LEARNING FUNDAMENTALS')

ml_foundations = [
    ('Q31: Bias-Variance Tradeoff', 'Bias is error from wrong assumptions (underfitting). Variance is error from sensitivity to noise (overfitting). Goal is to find the balance.'),
    ('Q32: Overfitting vs Underfitting', 'Overfitting: Low train error, High test error (too complex). Underfitting: High train error, High test error (too simple).'),
    ('Q33: L1 vs L2 Regularization', 'L1 (Lasso) adds absolute penalty and can do feature selection. L2 (Ridge) adds squared penalty. XGBoost uses both to control complexity.')
]

for q, a in ml_foundations:
    pdf.section_title(q)
    pdf.body_text(a)

# Section 13: Coding & SQL
pdf.add_page()
pdf.chapter_title('13. CODING & SQL CHALLENGES')
pdf.section_title('Python: Calculate Churn Rate')
pdf.body_text('def calculate_churn(statuses):\n    return (sum(statuses) / len(statuses)) * 100\n# Example: [0, 1, 0, 1] -> 50%')

pdf.section_title('SQL: High Value State Analysis')
pdf.body_text('SELECT state, COUNT(id), AVG(charge)\nFROM users GROUP BY state\nHAVING COUNT(id) > 50 ORDER BY AVG(charge) DESC;')

# Section 14: Questions for Interviewer
pdf.add_page()
pdf.chapter_title('14. QUESTIONS TO ASK THE INTERVIEWER')
pdf.bullet_point('How does the company handle retention for customers flagged by ML models?')
pdf.bullet_point('What is the typical deployment and monitoring cycle for models here?')
pdf.bullet_point('What are the biggest data quality challenges the team is currently facing?')
pdf.bullet_point('How is DS impact measured? Revenue vs Metric improvement?')

# Quick Reference
pdf.add_page()
pdf.chapter_title('QUICK REFERENCE CARD')
pdf.table_row(['Item', 'Value'], [80, 110], is_header=True)
pdf.table_row(['Best Model', 'XGBoost Classifier'], [80, 110])
pdf.table_row(['Accuracy / Recall', '98% / 87%'], [80, 110])
pdf.table_row(['Top Predictor', 'Service Calls (>=4)'], [80, 110])
pdf.table_row(['Preprocessing', 'StandardScaler + SMOTE'], [80, 110])
pdf.table_row(['Annual Savings', '$5.09 Million'], [80, 110])

pdf.ln(10)
pdf.set_font('Helvetica', 'B', 14)
pdf.set_text_color(26, 95, 122)
pdf.cell(0, 10, 'GOOD LUCK WITH YOUR INTERVIEW!', align='C')

# Save PDF
pdf.output('INTERVIEW_PREPARATION_GUIDE.pdf')
print('PDF created successfully!')
print('File: INTERVIEW_PREPARATION_GUIDE.pdf')

"""
Comprehensive Project Documentation PDF Generator
Creates a detailed PDF with all project sections, visualizations, and explanations
"""

from fpdf import FPDF, XPos, YPos
from pathlib import Path
import os

class ComprehensivePDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, 'Telecom Customer Churn Prediction - Comprehensive Documentation', 
                     align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(3)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')
    
    def chapter_title(self, num, title):
        self.set_font('Helvetica', 'B', 16)
        self.set_fill_color(26, 95, 122)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, f'{num}. {title}', fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)
    
    def section_title(self, title):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(46, 134, 171)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)
    
    def body_text(self, text):
        self.set_font('Helvetica', '', 11)
        self.set_text_color(51, 51, 51)
        text = text.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 6, text)
        self.ln(2)
    
    def bullet_point(self, text, indent=10):
        self.set_font('Helvetica', '', 11)
        self.set_text_color(51, 51, 51)
        text = text.encode('latin-1', 'replace').decode('latin-1')
        self.set_x(self.l_margin + indent)
        self.cell(5, 6, chr(149))  # Bullet character
        self.multi_cell(0, 6, f'  {text}')
    
    def add_image_with_caption(self, image_path, caption, width=180):
        if os.path.exists(image_path):
            x = (210 - width) / 2  # Center the image
            self.image(image_path, x=x, w=width)
            self.ln(3)
            self.set_font('Helvetica', 'I', 10)
            self.set_text_color(100, 100, 100)
            self.multi_cell(0, 5, caption, align='C')
            self.ln(5)
    
    def add_table(self, headers, data, col_widths):
        # Header
        self.set_font('Helvetica', 'B', 10)
        self.set_fill_color(26, 95, 122)
        self.set_text_color(255, 255, 255)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 8, header, border=1, fill=True, align='C')
        self.ln()
        
        # Data rows
        self.set_font('Helvetica', '', 10)
        self.set_text_color(51, 51, 51)
        fill = False
        for row in data:
            self.set_fill_color(245, 245, 245)
            for i, cell in enumerate(row):
                cell_text = str(cell).encode('latin-1', 'replace').decode('latin-1')
                self.cell(col_widths[i], 7, cell_text, border=1, fill=fill, align='C')
            self.ln()
            fill = not fill
        self.ln(3)

# Create PDF
pdf = ComprehensivePDF()
pdf.set_auto_page_break(auto=True, margin=15)
img_dir = Path('pdf_images')

# ==================== TITLE PAGE ====================
pdf.add_page()
pdf.set_font('Helvetica', 'B', 28)
pdf.set_text_color(26, 95, 122)
pdf.ln(50)
pdf.cell(0, 15, 'TELECOM CUSTOMER', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 15, 'CHURN PREDICTION', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(15)
pdf.set_font('Helvetica', '', 18)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 10, 'Comprehensive Project Documentation', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(20)

# Highlights box
pdf.set_line_width(1)
pdf.set_draw_color(26, 95, 122)
pdf.rect(30, pdf.get_y(), 150, 60)
pdf.ln(10)
pdf.set_font('Helvetica', 'B', 14)
pdf.set_text_color(26, 95, 122)
pdf.cell(0, 8, 'Project Highlights', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', 12)
pdf.set_text_color(51, 51, 51)
pdf.cell(0, 7, 'Dataset: 3,333 customers | 19 features', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 7, 'Best Model: XGBoost Classifier', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 7, 'Accuracy: 98% | Recall: 87%', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 7, 'Potential Annual Savings: $5+ Million', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# ==================== SECTION 1: PROJECT OBJECTIVE ====================
pdf.add_page()
pdf.chapter_title('1', 'PROJECT OBJECTIVE & BUSINESS PROBLEM')

pdf.section_title('1.1 What is Customer Churn?')
pdf.body_text('Customer churn, also known as customer attrition, occurs when customers stop doing business with a company or discontinue their subscription to a service. In the telecommunications industry, churn is a critical business metric that directly impacts revenue and profitability.')

pdf.section_title('1.2 Why is Churn Prediction Important?')
pdf.body_text('The telecommunication industry faces significant challenges with customer churn:')
pdf.bullet_point('Average annual churn rate: 10-25%')
pdf.bullet_point('Revenue loss per churned customer: $60-80 per month')
pdf.bullet_point('Customer acquisition cost is 5x higher than retention cost')
pdf.bullet_point('Lost customer lifetime value can exceed $1,000')
pdf.ln(3)

pdf.section_title('1.3 Business Impact Example')
pdf.body_text('Consider a telecom company with 1 million customers:')
pdf.bullet_point('15% annual churn = 150,000 customers lost')
pdf.bullet_point('Average revenue per customer = $65/month')
pdf.bullet_point('Annual revenue loss = 150,000 x $65 x 12 = $117 MILLION')
pdf.ln(3)

pdf.section_title('1.4 Project Objective')
pdf.set_font('Helvetica', 'BI', 11)
pdf.set_fill_color(255, 250, 235)
pdf.multi_cell(0, 7, 'Build a predictive machine learning model that identifies at-risk customers BEFORE they churn, enabling proactive retention strategies such as personalized discounts, dedicated support, or loyalty rewards programs.', fill=True)
pdf.ln(3)

pdf.section_title('1.5 Expected Outcomes')
pdf.bullet_point('Predict customer churn with high accuracy (>95%)')
pdf.bullet_point('Identify key factors driving churn behavior')
pdf.bullet_point('Enable data-driven retention strategies')
pdf.bullet_point('Reduce customer attrition by 30-50%')
pdf.bullet_point('Save millions in revenue annually')

# ==================== SECTION 2: DATASET OVERVIEW ====================
pdf.add_page()
pdf.chapter_title('2', 'DATASET OVERVIEW & DESCRIPTION')

pdf.section_title('2.1 Dataset Summary')
pdf.add_table(
    ['Property', 'Value'],
    [
        ['Total Customers', '3,333'],
        ['Total Features', '19 (+ 1 target)'],
        ['Missing Values', 'None (0%)'],
        ['Duplicate Records', 'None'],
        ['Data Type', 'Numerical & Binary'],
        ['Target Variable', 'churn (0=Stay, 1=Leave)']
    ],
    [80, 110]
)

pdf.section_title('2.2 Feature Categories')
pdf.body_text('The dataset contains four main categories of features:')
pdf.ln(2)

pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 7, '1. Account Information:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', 10)
pdf.bullet_point('account_length: Number of days as a customer', indent=15)
pdf.bullet_point('international_plan: Has international calling (0/1)', indent=15)
pdf.bullet_point('voice_mail_plan: Has voicemail service (0/1)', indent=15)
pdf.bullet_point('voice_mail_messages: Number of voicemail messages', indent=15)
pdf.ln(2)

pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 7, '2. Usage Patterns:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', 10)
pdf.bullet_point('day_mins, evening_mins, night_mins, international_mins', indent=15)
pdf.bullet_point('day_calls, evening_calls, night_calls, international_calls', indent=15)
pdf.ln(2)

pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 7, '3. Billing Information:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', 10)
pdf.bullet_point('day_charge, evening_charge, night_charge, international_charge', indent=15)
pdf.bullet_point('total_charge: Total monthly bill ($)', indent=15)
pdf.ln(2)

pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 7, '4. Customer Service:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', 10)
pdf.bullet_point('customer_service_calls: Number of calls to support', indent=15)
pdf.ln(2)

pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 7, '5. Target Variable:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', 10)
pdf.bullet_point('churn: 0 = Customer stayed, 1 = Customer left', indent=15)

# ==================== SECTION 3: STATISTICAL SUMMARY ====================
pdf.add_page()
pdf.chapter_title('3', 'STATISTICAL SUMMARY')

pdf.section_title('3.1 Key Statistics')
pdf.body_text('Below are the descriptive statistics for the most important numerical features:')
pdf.ln(2)

pdf.add_table(
    ['Feature', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
    [
        ['total_charge', '59.8', '56.3', '16.3', '18.3', '107.2'],
        ['day_mins', '179.8', '179.4', '54.4', '0', '359.4'],
        ['customer_service_calls', '1.6', '1.0', '1.3', '0', '9'],
        ['account_length', '101.1', '101.0', '39.8', '1', '243'],
        ['voice_mail_messages', '8.1', '0', '13.7', '0', '51'],
        ['international_mins', '10.2', '10.3', '2.8', '0', '20'],
    ],
    [50, 25, 25, 25, 20, 20]
)

pdf.section_title('3.2 Target Variable Distribution')
pdf.body_text('The churn distribution shows a significant class imbalance:')
pdf.ln(2)

pdf.add_table(
    ['Class', 'Count', 'Percentage'],
    [
        ['Retained (0)', '2,850', '85.5%'],
        ['Churned (1)', '483', '14.5%']
    ],
    [60, 65, 65]
)

pdf.body_text('This imbalance is realistic for the telecom industry but requires special handling during model training to prevent bias toward the majority class.')

# ==================== SECTION 4: EDA WITH DIAGRAMS ====================
pdf.add_page()
pdf.chapter_title('4', 'EXPLORATORY DATA ANALYSIS (EDA)')

pdf.section_title('4.1 Target Variable Distribution')
pdf.body_text('The pie chart below shows that approximately 85.5% of customers are retained while 14.5% churn. This class imbalance is typical in churn prediction scenarios.')
pdf.ln(2)
pdf.add_image_with_caption(
    str(img_dir / 'churn_distribution.png'),
    'Figure 4.1: Customer Churn Distribution - Shows the proportion of retained vs churned customers',
    width=120
)

pdf.section_title('4.2 Feature Distributions by Churn Status')
pdf.body_text('The histograms reveal how different features vary between retained (blue) and churned (red) customers. Key observations:')
pdf.bullet_point('Churned customers tend to have higher total charges')
pdf.bullet_point('Day minutes show higher usage among churned customers')
pdf.bullet_point('Customer service calls are significantly higher for churned customers')
pdf.ln(2)
pdf.add_image_with_caption(
    str(img_dir / 'feature_distributions.png'),
    'Figure 4.2: Feature Distributions - Overlaid histograms showing patterns between retained and churned customers',
    width=180
)

# Correlation Heatmap
pdf.add_page()
pdf.section_title('4.3 Correlation Analysis')
pdf.body_text('The correlation heatmap reveals relationships between features:')
pdf.bullet_point('Strong positive correlation (0.99) between minutes and charges (expected)')
pdf.bullet_point('Moderate positive correlation between total_charge and churn (+0.23)')
pdf.bullet_point('Customer service calls show positive correlation with churn (+0.21)')
pdf.bullet_point('International plan has +0.26 correlation with churn')
pdf.ln(2)
pdf.add_image_with_caption(
    str(img_dir / 'correlation_heatmap.png'),
    'Figure 4.3: Correlation Matrix - Red indicates positive correlation, Blue indicates negative correlation',
    width=175
)

# Box Plots
pdf.add_page()
pdf.section_title('4.4 Outlier Analysis with Box Plots')
pdf.body_text('Box plots help identify outliers and compare distributions between retained and churned customers:')
pdf.bullet_point('Churned customers have higher median total charges')
pdf.bullet_point('Customer service calls show clear separation - churned customers call more')
pdf.bullet_point('Outliers are kept in the dataset as they represent real high-value or dissatisfied customers')
pdf.ln(2)
pdf.add_image_with_caption(
    str(img_dir / 'box_plots.png'),
    'Figure 4.4: Box Plot Analysis - Box contains 50% of data, line is median, dots are outliers',
    width=180
)

# Service Calls Analysis
pdf.add_page()
pdf.section_title('4.5 Customer Service Calls vs Churn Rate')
pdf.body_text('This is the MOST IMPORTANT finding from EDA:')
pdf.bullet_point('Customers with 0-2 calls: ~10% churn rate (Low Risk)')
pdf.bullet_point('Customers with 3 calls: ~20% churn rate (Medium Risk)')
pdf.bullet_point('Customers with 4+ calls: 45-80% churn rate (HIGH RISK)')
pdf.ln(2)
pdf.set_font('Helvetica', 'BI', 11)
pdf.set_fill_color(255, 235, 235)
pdf.multi_cell(0, 6, 'KEY INSIGHT: Each service call indicates unresolved frustration. After 4 calls, customers are HIGHLY likely to churn. This is our #1 predictor!', fill=True)
pdf.ln(3)
pdf.add_image_with_caption(
    str(img_dir / 'service_calls_churn.png'),
    'Figure 4.5: Churn Rate by Service Calls - Red bars indicate high-risk levels (>20% churn)',
    width=160
)

# International Plan Analysis
pdf.add_page()
pdf.section_title('4.6 International Plan Impact')
pdf.body_text('Customers with international plans show 3x higher churn:')
pdf.bullet_point('No Plan: 11% churn rate')
pdf.bullet_point('Has Plan: 28% churn rate')
pdf.ln(2)
pdf.body_text('Possible reasons: High international rates, poor call quality, or better competitor offers.')
pdf.ln(2)
pdf.add_image_with_caption(
    str(img_dir / 'international_plan_churn.png'),
    'Figure 4.6: International Plan Impact - Plan holders churn at significantly higher rates',
    width=140
)

# ==================== SECTION 5: FEATURE ENGINEERING ====================
pdf.add_page()
pdf.chapter_title('5', 'FEATURE ENGINEERING & SELECTION')

pdf.section_title('5.1 Feature Selection Strategy')
pdf.body_text('We used Random Forest feature importance to identify the top 10 most predictive features out of 19 total features. This approach:')
pdf.bullet_point('Reduces dimensionality and training time')
pdf.bullet_point('Removes noise from less important features')
pdf.bullet_point('Improves model generalization')
pdf.bullet_point('Provides interpretability')
pdf.ln(3)

pdf.section_title('5.2 Top 10 Selected Features')
pdf.add_image_with_caption(
    str(img_dir / 'feature_importance.png'),
    'Figure 5.1: Feature Importance Rankings - Top features contribute most to churn prediction',
    width=160
)

pdf.body_text('The top 3 features account for over 55% of predictive power:')
pdf.bullet_point('Total Charge (25%): Overall billing impact')
pdf.bullet_point('Customer Service Calls (18%): Frustration indicator')
pdf.bullet_point('International Plan (15%): Key risk factor')

# ==================== SECTION 6: DATA PREPROCESSING ====================
pdf.add_page()
pdf.chapter_title('6', 'DATA PREPROCESSING')

pdf.section_title('6.1 Standard Scaling')
pdf.body_text('StandardScaler was applied to normalize all features to have mean=0 and standard deviation=1.')
pdf.ln(2)
pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 6, 'Formula: z = (x - mean) / std_dev', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(2)
pdf.set_font('Helvetica', '', 11)
pdf.body_text('Why scaling is necessary:')
pdf.bullet_point('Features have different scales (day_mins: 0-400 vs charge: 0-5)')
pdf.bullet_point('Large-scale features can dominate the model')
pdf.bullet_point('Many ML algorithms perform better with normalized data')
pdf.ln(3)

pdf.section_title('6.2 Handling Class Imbalance with SMOTE')
pdf.body_text('SMOTE (Synthetic Minority Oversampling Technique) creates synthetic samples of the minority class by interpolating between existing minority samples.')
pdf.ln(2)

pdf.add_image_with_caption(
    str(img_dir / 'smote_comparison.png'),
    'Figure 6.1: SMOTE Balancing - Before: 85.5% vs 14.5%, After: 50% vs 50%',
    width=150
)

pdf.body_text('How SMOTE works:')
pdf.bullet_point('1. For each minority sample, find k-nearest neighbors (default k=5)')
pdf.bullet_point('2. Randomly select one neighbor')
pdf.bullet_point('3. Create new sample = original + random(0,1) x (neighbor - original)')
pdf.bullet_point('4. Repeat until classes are balanced')
pdf.ln(2)
pdf.body_text('Why SMOTE is better than simple duplication:')
pdf.bullet_point('Creates NEW synthetic points rather than exact copies')
pdf.bullet_point('Reduces overfitting risk')
pdf.bullet_point('Provides better model generalization')

# ==================== SECTION 7: TRAIN-TEST SPLIT ====================
pdf.add_page()
pdf.chapter_title('7', 'TRAIN-TEST SPLIT')

pdf.section_title('7.1 Split Configuration')
pdf.body_text('The dataset was split using an 80-20 ratio:')
pdf.ln(2)

pdf.add_table(
    ['Dataset', 'Samples', 'Percentage', 'Purpose'],
    [
        ['Training Set', '2,666', '80%', 'Model Learning'],
        ['Test Set', '667', '20%', 'Unbiased Evaluation']
    ],
    [45, 35, 35, 75]
)

pdf.section_title('7.2 Why 80-20 Split?')
pdf.bullet_point('Standard industry practice for balanced datasets')
pdf.bullet_point('80% provides sufficient data for model training')
pdf.bullet_point('20% provides statistically significant test set (667 samples)')
pdf.bullet_point('Test set includes 101 churned customers for reliable evaluation')
pdf.ln(2)
pdf.body_text('The split was performed AFTER SMOTE to ensure balanced classes in both sets.')

# ==================== SECTION 8: MODEL BUILDING ====================
pdf.add_page()
pdf.chapter_title('8', 'MODEL BUILDING')

pdf.section_title('8.1 Models Evaluated')
pdf.body_text('Two ensemble learning methods were compared:')
pdf.ln(2)

pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 7, '1. Random Forest Classifier (Bagging)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', 10)
pdf.bullet_point('Builds 100 decision trees independently in parallel', indent=10)
pdf.bullet_point('Each tree trained on random bootstrap samples', indent=10)
pdf.bullet_point('Final prediction = majority vote of all trees', indent=10)
pdf.bullet_point('Reduces variance, prevents overfitting', indent=10)
pdf.ln(2)

pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 7, '2. XGBoost Classifier (Boosting)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', 10)
pdf.bullet_point('Builds trees sequentially', indent=10)
pdf.bullet_point('Each tree corrects errors of previous trees', indent=10)
pdf.bullet_point('Final prediction = weighted sum of all trees', indent=10)
pdf.bullet_point('Built-in L1/L2 regularization prevents overfitting', indent=10)
pdf.ln(3)

pdf.section_title('8.2 Model Comparison Results')
pdf.add_image_with_caption(
    str(img_dir / 'model_comparison.png'),
    'Figure 8.1: Model Performance - XGBoost outperforms Random Forest on both metrics',
    width=160
)

pdf.section_title('8.3 Winner: XGBoost')
pdf.body_text('XGBoost was selected as the final model due to:')
pdf.bullet_point('Higher Accuracy: 98% vs 97%')
pdf.bullet_point('Better Recall: 87% vs 81% (catches 6% more churners)')
pdf.bullet_point('Fewer False Negatives: 13 vs 20 (7 fewer missed churners)')
pdf.bullet_point('Robust to outliers and imbalanced data')

# ==================== SECTION 9: MODEL EVALUATION ====================
pdf.add_page()
pdf.chapter_title('9', 'MODEL EVALUATION')

pdf.section_title('9.1 Confusion Matrix')
pdf.add_image_with_caption(
    str(img_dir / 'confusion_matrix.png'),
    'Figure 9.1: XGBoost Confusion Matrix - Darker blues indicate higher counts',
    width=130
)

pdf.section_title('9.2 Confusion Matrix Breakdown')
pdf.add_table(
    ['Cell', 'Count', 'Meaning'],
    [
        ['True Negatives (TN)', '566', 'Correctly predicted staying customers'],
        ['False Positives (FP)', '0', 'Wrongly predicted as churners'],
        ['False Negatives (FN)', '13', 'Missed churners (predicted stay, actually churned)'],
        ['True Positives (TP)', '88', 'Correctly predicted churners']
    ],
    [55, 30, 105]
)

pdf.section_title('9.3 Classification Metrics')
pdf.body_text('Accuracy = (TP + TN) / Total = (88 + 566) / 667 = 98%')
pdf.body_text('Precision = TP / (TP + FP) = 88 / (88 + 0) = 100%')
pdf.body_text('Recall = TP / (TP + FN) = 88 / (88 + 13) = 87%')
pdf.body_text('F1-Score = 2 x (Precision x Recall) / (Precision + Recall) = 0.93')
pdf.ln(3)

pdf.section_title('9.4 Why Recall is More Important')
pdf.body_text('In churn prediction, False Negatives are more costly than False Positives:')
pdf.ln(2)

pdf.add_table(
    ['Error Type', 'Business Impact', 'Cost'],
    [
        ['False Positive', 'Give discount to loyal customer', 'Small (~$10-20)'],
        ['False Negative', 'Customer leaves without intervention', 'Large (~$780/year)']
    ],
    [50, 75, 65]
)

pdf.body_text('Therefore, we prioritize RECALL (catching churners) over Precision to minimize revenue loss.')

# ==================== SECTION 10: INTERVIEW QUESTIONS ====================
pdf.add_page()
pdf.chapter_title('10', 'INTERVIEW QUESTIONS & ANSWERS')

pdf.section_title('Q1: Explain your project in 2 minutes')
pdf.body_text('I built a customer churn prediction system for a telecom company using machine learning. The dataset had 3,333 customers with 19 features. After EDA, I discovered that service calls and international plans were major churn indicators. I handled class imbalance with SMOTE, selected top 10 features using Random Forest, and compared two models. XGBoost achieved 98% accuracy and 87% recall, catching most at-risk customers. The model can save the company $5+ million annually through proactive retention.')

pdf.section_title('Q2: What was your biggest challenge?')
pdf.body_text('The class imbalance (85.5% vs 14.5%). Without handling, the model would just predict "no churn" and get 85% accuracy but miss all churners. I solved this with SMOTE, which created synthetic minority samples, balancing the classes to 50-50. This dramatically improved the model\' s ability to detect churners.')

pdf.section_title('Q3: How did you handle outliers?')
pdf.body_text('I kept them. The outliers (high service calls, high usage) represent exactly the dissatisfied customers we want to predict. Removing them would remove critical signal. XGBoost is also robust to outliers due to its tree-based nature.')

pdf.section_title('Q4: Explain SMOTE technically')
pdf.body_text('SMOTE: 1) For each minority sample, find k-nearest neighbors (k=5), 2) Randomly select one neighbor, 3) Create new sample = original + random(0,1) x (neighbor - original), 4) Repeat until balanced. Unlike duplication, this creates NEW synthetic points in feature space.')

pdf.section_title('Q5: Why XGBoost over Random Forest?')
pdf.body_text('XGBoost uses gradient boosting - builds trees sequentially where each corrects previous errors. It achieved 98% accuracy vs 97% for Random Forest, and more importantly 87% recall vs 81%, catching 6% more churners. It also has built-in L1/L2 regularization.')

pdf.section_title('Q6: What is your business recommendation?')
pdf.body_text('1) Flag customers after 2nd service call, 2) Assign dedicated support after 3rd call, 3) Offer 15-20% retention discount after 4th call, 4) Review international plan pricing (28% churn vs 11%), 5) Offer free voicemail trials to at-risk customers (reduces churn 50%).')

pdf.section_title('Q7: How would you deploy this?')
pdf.body_text('1) Save model with joblib/pickle, 2) Create REST API with Flask/FastAPI, 3) Containerize with Docker, 4) Deploy on AWS/GCP with auto-scaling, 5) Set up monitoring for model drift, 6) Implement A/B testing for retention strategies, 7) Schedule weekly retraining with new data.')

pdf.section_title('Q8: What would you improve?')
pdf.body_text('1) K-fold cross-validation instead of single split, 2) Hyperparameter tuning with GridSearchCV, 3) Add SHAP values for model interpretability, 4) Engineer time-based features (usage trends), 5) Implement real-time prediction pipeline, 6) Test ensemble of XGBoost + LightGBM.')

# ==================== SECTION 11: STORYTELLING ====================
pdf.add_page()
pdf.chapter_title('11', 'STORYTELLING FOR NON-TECHNICAL AUDIENCE')

pdf.section_title('The Problem')
pdf.body_text('Imagine you run a telecom company with 1 million customers. Every month, 15,000 customers cancel their service. Each lost customer costs you $65/month, totaling $117 million in lost revenue annually. Traditional methods only react AFTER customers leave. What if you could predict WHO will leave BEFORE they do?')

pdf.section_title('The Solution')
pdf.body_text('We built an AI system that analyzes customer behavior patterns to identify who is likely to cancel. Think of it like a weather forecast - instead of predicting rain, we predict customer churn.')

pdf.section_title('How It Works (Simple Explanation)')
pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 7, 'Step 1: Collect Data', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', 10)
pdf.body_text('We looked at 3,333 customers and collected information like monthly bill, call usage, plan type, and how many times they called customer service.')

pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 7, 'Step 2: Find Patterns', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', 10)
pdf.body_text('We discovered surprising patterns: Customers who called support 4+ times had a 45% chance of leaving. Customers with international plans were 3x more likely to cancel. Customers with voicemail were 50% less likely to leave.')

pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 7, 'Step 3: Train the AI', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', 10)
pdf.body_text('We taught a computer program to recognize these patterns. Like showing a child 1,000 pictures of cats and dogs until they can tell them apart, we showed our AI 2,666 customer records until it learned to spot potential churners.')

pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 7, 'Step 4: Test Accuracy', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', 10)
pdf.body_text('We tested the AI on 667 new customers it had never seen. It correctly predicted 98% of outcomes, catching 87% of customers who actually left.')

pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 7, 'Step 5: Take Action', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', 10)
pdf.body_text('Now, every day, the AI scans all customers and flags high-risk ones. The retention team contacts them with special offers, dedicated support, or plan adjustments.')

pdf.section_title('The Results')
pdf.body_text('If we deploy this system to 100,000 customers:')
pdf.bullet_point('AI identifies 13,050 of the 15,000 at-risk customers (87%)')
pdf.bullet_point('Retention team successfully saves 50% of contacted customers')
pdf.bullet_point('6,525 customers retained who would have left')
pdf.bullet_point('Annual savings: $5.09 MILLION')
pdf.ln(3)

pdf.section_title('The Magic Behind It')
pdf.body_text('Think of the AI like a detective with a checklist:')
pdf.bullet_point('Has customer called support more than 3 times? +30 risk points')
pdf.bullet_point('Does customer have international plan? +20 risk points')
pdf.bullet_point('Is monthly bill above $70? +15 risk points')
pdf.bullet_point('Has voicemail? -10 risk points')
pdf.ln(2)
pdf.body_text('If risk score > 50 points, flag as "High Risk - Contact Immediately"')

pdf.section_title('Why This Matters')
pdf.body_text('Unlike traditional methods that wait for customers to complain or leave, this AI system is PROACTIVE. It identifies problems before customers even think about canceling. It transforms customer retention from reactive firefighting to strategic prevention.')

pdf.section_title('Real-World Impact')
pdf.body_text('Consider Maria, a customer with an international plan and high bills. She called support 3 times about charges. Traditional approach: wait to see if she leaves. AI approach: Flag Maria immediately, offer a better international plan with 20% discount. Maria stays, company keeps $780/year revenue.')

# ==================== FINAL PAGE ====================
pdf.add_page()
pdf.ln(80)
pdf.set_font('Helvetica', 'B', 22)
pdf.set_text_color(26, 95, 122)
pdf.cell(0, 15, 'Thank You', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(10)
pdf.set_font('Helvetica', '', 14)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 10, 'Shashank R', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(0, 10, 'Data Scientist', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# Save PDF
output_file = 'COMPREHENSIVE_PROJECT_DOCUMENTATION.pdf'
pdf.output(output_file)
print(f'PDF created successfully: {output_file}')
print(f'Total pages: {pdf.page_no()}')

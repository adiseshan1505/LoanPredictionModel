<h1 align="center">Loan Prediction Model</h1>
<h2>Tech Stack:- </h2>
<li>
  <ul>Used Python as the main programming software.</ul>
  <ul>Imported a .csv file from Kaggle:- Loan_default.csv</ul>
  <ul>Cleaned, maipulated the data using pandas library.</ul>
  <ul>From the main dataset I have extracted only a few(necessary) columns like Default (0 means credible(borrower repays successfully) or 1(borrower fails to repay)), LoanAmount, Income, DTIRatio, CreditScore</ul>
  <ul>That cleaned data I have imported it to a file cleaned_data.csv</ul>
  <ul>Using matplotlib I have plotted a histogram(using seaborn) that explains the Loan Amount Distribution</ul>
  <ul>Then using scikit-learn libraries I have created a simple ML model (trains for 80% and tests for 20%) that returns accuracy value for the default column and the overall accuracy value as well</ul>
  <ul>A visualized report has also been created with the help of Power-BI.</ul>
  <ul>The report includes DAX expressions expressed in form of flash cards. A pie chart explaining count of default, a slicer for credit score(you can change the range values and manipulate all reading s in the report).</ul>
</li>

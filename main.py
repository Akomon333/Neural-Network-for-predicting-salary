import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
# AVAILABLE WORK YEARS 2020 - 2023
def int_to_company_location(number):
    countrys_indexes = ['Spain', 'United States', 'Canada', 'Germany', 'United Kingdom', 'Nigeria', 'India',
                 'Hong Kong', 'Netherlands', 'Switzerland', 'Central African Republic', 'France',
                 'Finland', 'Ukraine', 'Ireland', 'Israel', 'Ghana', 'Colombia', 'Singapore',
                 'Australia', 'Sweden', 'Slovenia', 'Mexico', 'Brazil', 'Portugal', 'Russia',
                 'Thailand', 'Croatia', 'Argentina', 'Lithuania', 'American Samoa', 'Costa Rica',
                 'Iran', 'Bahamas', 'Hungary', 'Austria', 'Slovakia', 'Czech Republic', 'Turkey',
                 'Puerto Rico', 'Denmark', 'Bolivia', 'Philippines', 'Belgium', 'Indonesia',
                 'Egypt', 'United Arab Emirates', 'Luxembourg', 'Malaysia', 'Honduras', 'Japan',
                 'Algeria', 'Iraq', 'China', 'New Zealand', 'Chile', 'Moldova', 'Malta']

    
    if 0 <= number < len(countrys_indexes):
        return countrys_indexes[number - 1] # I forgor to include number 0 in company_location
    else:
        return "Invalid index" 
def int_to_experience_level(number):
    levels = ['Senior', 'Middle', 'Entry level','Executive'] 
    if 0 <= number < len(levels):
        return levels[number] 
    else:
        return "Invalid index" 
def int_to_work_type(number):
    types = ['Full time', 'Contractor', 'Freelance', 'Part time']
    if 0 <= number < len(types):
        return types[number] 
    else:
        return "Invalid index" 

def int_to_job_title(number):
    jobs = ['Principal Data Scientist', 'ML Engineer', 'Data Scientist', 'Applied Scientist', 'Data Analyst', 'Data Modeler', 'Research Engineer', 'Analytics Engineer', 'Business Intelligence Engineer', 'Machine Learning Engineer', 'Data Strategist', 'Data Engineer', 'Computer Vision Engineer', 'Data Quality Analyst', 'Compliance Data Analyst', 'Data Architect', 'Applied Machine Learning Engineer', 'AI Developer', 'Research Scientist', 'Data Analytics Manager', 'Business Data Analyst', 'Applied Data Scientist', 'Staff Data Analyst', 'ETL Engineer', 'Data DevOps Engineer', 'Head of Data', 'Data Science Manager', 'Data Manager', 'Machine Learning Researcher', 'Big Data Engineer', 'Data Specialist', 'Lead Data Analyst', 'BI Data Engineer', 'Director of Data Science', 'Machine Learning Scientist', 'MLOps Engineer', 'AI Scientist', 'Autonomous Vehicle Technician', 'Applied Machine Learning Scientist', 'Lead Data Scientist', 'Cloud Database Engineer', 'Financial Data Analyst', 'Data Infrastructure Engineer', 'Software Data Engineer', 'AI Programmer', 'Data Operations Engineer', 'BI Developer', 'Data Science Lead', 'Deep Learning Researcher', 'BI Analyst', 'Data Science Consultant', 'Data Analytics Specialist', 'Machine Learning Infrastructure Engineer', 'BI Data Analyst', 'Head of Data Science', 'Insight Analyst', 'Deep Learning Engineer', 'Machine Learning Software Engineer', 'Big Data Architect', 'Product Data Analyst', 'Computer Vision Software Engineer', 'Azure Data Engineer', 'Marketing Data Engineer', 'Data Analytics Lead', 'Data Lead', 'Data Science Engineer', 'Machine Learning Research Engineer', 'NLP Engineer', 'Manager Data Management', 'Machine Learning Developer', '3D Computer Vision Researcher', 'Principal Machine Learning Engineer', 'Data Analytics Engineer', 'Data Analytics Consultant', 'Data Management Specialist', 'Data Science Tech Lead', 'Data Scientist Lead', 'Cloud Data Engineer', 'Data Operations Analyst', 'Marketing Data Analyst', 'Power BI Developer', 'Product Data Scientist', 'Principal Data Architect', 'Machine Learning Manager', 'Lead Machine Learning Engineer', 'ETL Developer', 'Cloud Data Architect', 'Lead Data Engineer', 'Head of Machine Learning', 'Principal Data Analyst', 'Principal Data Engineer', 'Staff Data Scientist', 'Finance Data Analyst']
    if 0 <= number < len(jobs):
        return jobs[number] 
    else:
        return "Invalid index"   
def int_to_company_size(number):
    sizes = ['Large', 'Small', 'Medium']
    if 0 <= number < len(sizes):
        return sizes[number] 
    else:
        return "Invalid index" 
def int_to_year(number):
    sizes = ["2023", "2022", "2020", "2021"]
    if 0 <= number < len(sizes):
        return sizes[number] 
    else:
        print("invalid year")
        return "Invalid index" 
def int_to_salary(number):
    return number * 100000 
def company_location_to_index(location):
    countrys_indexes = ['Spain', 'United States', 'Canada', 'Germany', 'United Kingdom', 'Nigeria', 'India',
                        'Hong Kong', 'Netherlands', 'Switzerland', 'Central African Republic', 'France',
                        'Finland', 'Ukraine', 'Ireland', 'Israel', 'Ghana', 'Colombia', 'Singapore',
                        'Australia', 'Sweden', 'Slovenia', 'Mexico', 'Brazil', 'Portugal', 'Russia',
                        'Thailand', 'Croatia', 'Argentina', 'Lithuania', 'American Samoa', 'Costa Rica',
                        'Iran', 'Bahamas', 'Hungary', 'Austria', 'Slovakia', 'Czech Republic', 'Turkey',
                        'Puerto Rico', 'Denmark', 'Bolivia', 'Philippines', 'Belgium', 'Indonesia',
                        'Egypt', 'United Arab Emirates', 'Luxembourg', 'Malaysia', 'Honduras', 'Japan',
                        'Algeria', 'Iraq', 'China', 'New Zealand', 'Chile', 'Moldova', 'Malta']
    
    if location in countrys_indexes:
        return countrys_indexes.index(location) + 1  # Add 1 to match original index
    else:
        print("invalid location")
        return "Invalid location"

def experience_level_to_index(level):
    levels = ['Senior', 'Middle', 'Entry level', 'Executive']
    if level in levels:
        return levels.index(level)
    else:
        print("invalid level to index")
        return "Invalid level"

def work_type_to_index(work_type):
    types = ['Full time', 'Contractor', 'Freelance', 'Part time']
    if work_type in types:
        return types.index(work_type)
    else:
        print("invalid work type to index")
        return "Invalid work type"

def job_title_to_index(title):
    jobs = ['Principal Data Scientist', 'ML Engineer', 'Data Scientist', 'Applied Scientist', 'Data Analyst', 'Data Modeler', 'Research Engineer', 'Analytics Engineer', 'Business Intelligence Engineer', 'Machine Learning Engineer', 'Data Strategist', 'Data Engineer', 'Computer Vision Engineer', 'Data Quality Analyst', 'Compliance Data Analyst', 'Data Architect', 'Applied Machine Learning Engineer', 'AI Developer', 'Research Scientist', 'Data Analytics Manager', 'Business Data Analyst', 'Applied Data Scientist', 'Staff Data Analyst', 'ETL Engineer', 'Data DevOps Engineer', 'Head of Data', 'Data Science Manager', 'Data Manager', 'Machine Learning Researcher', 'Big Data Engineer', 'Data Specialist', 'Lead Data Analyst', 'BI Data Engineer', 'Director of Data Science', 'Machine Learning Scientist', 'MLOps Engineer', 'AI Scientist', 'Autonomous Vehicle Technician', 'Applied Machine Learning Scientist', 'Lead Data Scientist', 'Cloud Database Engineer', 'Financial Data Analyst', 'Data Infrastructure Engineer', 'Software Data Engineer', 'AI Programmer', 'Data Operations Engineer', 'BI Developer', 'Data Science Lead', 'Deep Learning Researcher', 'BI Analyst', 'Data Science Consultant', 'Data Analytics Specialist', 'Machine Learning Infrastructure Engineer', 'BI Data Analyst', 'Head of Data Science', 'Insight Analyst', 'Deep Learning Engineer', 'Machine Learning Software Engineer', 'Big Data Architect', 'Product Data Analyst', 'Computer Vision Software Engineer', 'Azure Data Engineer', 'Marketing Data Engineer', 'Data Analytics Lead', 'Data Lead', 'Data Science Engineer', 'Machine Learning Research Engineer', 'NLP Engineer', 'Manager Data Management', 'Machine Learning Developer', '3D Computer Vision Researcher', 'Principal Machine Learning Engineer', 'Data Analytics Engineer', 'Data Analytics Consultant', 'Data Management Specialist', 'Data Science Tech Lead', 'Data Scientist Lead', 'Cloud Data Engineer', 'Data Operations Analyst', 'Marketing Data Analyst', 'Power BI Developer', 'Product Data Scientist', 'Principal Data Architect', 'Machine Learning Manager', 'Lead Machine Learning Engineer', 'ETL Developer', 'Cloud Data Architect', 'Lead Data Engineer', 'Head of Machine Learning', 'Principal Data Analyst', 'Principal Data Engineer', 'Staff Data Scientist', 'Finance Data Analyst']
    if title in jobs:
        return jobs.index(title)
    else:
        print("invalid job title index")
        return "Invalid job title"

def company_size_to_index(size):
    sizes = ['Large', 'Small', 'Medium']
    if size in sizes:
        return sizes.index(size)
    else:
        print("invalid company size")
        return "Invalid company size"

def year_to_index(year):
    sizes = ["2023", "2022", "2020", "2021"]
    if year in sizes:
        return sizes.index(year)
    else:
        print("invalid year")
        return "Invalid year"

def salary_to_index(salary):
    try:
        return int(salary) // 100000  # Dividing salary by 100,000 to get the index
    except ValueError:
        return "Invalid salary"



file = pd.read_csv("./ds_salaries.csv")


# 6 inputs 1 output(salary)
# output salary_in_usd
#inputs work_year, experience_level, employment_type, job_title, company_location, company_size
class Model(nn.Module):
    def __init__(self, inputs=6,h1=20,h2=40,h3=60,h4=80,h5=90,output_features=1): # fc = fully connected layer(connectin layers)
        super().__init__()
        self.fc1 = nn.Linear(inputs,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.fc3 = nn.Linear(h2,h3)
        self.fc4 = nn.Linear(h3,h4)
        self.fc5 = nn.Linear(h4,h5)
        self.out = nn.Linear(h5,output_features)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.out(x)

        return x
    

torch.manual_seed(41)

model = Model()
model.load_state_dict(torch.load("Tech_Salary_Prediction_o1"))


X = file.drop('salary_in_usd',axis=1) # Features
y = file['salary_in_usd'] # Salaries
X = X.values
y = y.values


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=41)


X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train).view(-1, 1).float()
y_test = torch.tensor(y_test).view(-1, 1).float()

criterion = nn.SmoothL1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100

for i in range(epochs):
    y_prediction = model(X_train)  # don't call .forward()

    loss = criterion(y_prediction, y_train)

    print(f'Epoch: {i} - Loss: {loss.item():.4f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        torch.save(model.state_dict(), 'Tech_Salary_Prediction_o1')




torch.save(model.state_dict(), 'Tech_Salary_Prediction_o1')

with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)
    print(f'Testing loss: {loss}')


#'work_year' 'experience_level' 'employment_type' 'job_title' 'company_location' 'company_size'
year = 2020
explevel = "Senior" 
employtype = "Full time"
job = "ML Engineer"
location = "United States"
size = "Large"
input_features = [
    year_to_index(str(year)),
    experience_level_to_index(explevel),
    work_type_to_index(employtype),
    job_title_to_index(job),
    company_location_to_index(location),
    company_size_to_index(size)
]
input_tensor = torch.tensor(input_features, dtype=torch.float32).view(1, -1)

with torch.no_grad():
    salary_prediction = model(input_tensor) 
    salary_prediction = salary_prediction.item()
    print(int_to_salary(salary_prediction))
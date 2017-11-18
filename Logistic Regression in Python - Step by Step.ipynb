{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Susan\\Anaconda3\\lib\\site-packages\\sklearn\\cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.\n",
      "  \"This module will be removed in 0.20.\", DeprecationWarning)\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn import preprocessing\n",
    "import matplotlib.pyplot as plt \n",
    "plt.rc(\"font\", size=14)\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.cross_validation import train_test_split\n",
    "import seaborn as sns\n",
    "sns.set(style=\"white\")\n",
    "sns.set(style=\"whitegrid\", color_codes=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data\n",
    "\n",
    "The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe (1/0) a term deposit (variable y)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This dataset provides the customer information. It includes 41188 records and 21 fields."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(41188, 21)\n",
      "['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed', 'y']\n"
     ]
    }
   ],
   "source": [
    "data = pd.read_csv('bank.csv', header=0)\n",
    "data = data.dropna()\n",
    "print(data.shape)\n",
    "print(list(data.columns))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>job</th>\n",
       "      <th>marital</th>\n",
       "      <th>education</th>\n",
       "      <th>default</th>\n",
       "      <th>housing</th>\n",
       "      <th>loan</th>\n",
       "      <th>contact</th>\n",
       "      <th>month</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>...</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "      <th>poutcome</th>\n",
       "      <th>emp_var_rate</th>\n",
       "      <th>cons_price_idx</th>\n",
       "      <th>cons_conf_idx</th>\n",
       "      <th>euribor3m</th>\n",
       "      <th>nr_employed</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>44</td>\n",
       "      <td>blue-collar</td>\n",
       "      <td>married</td>\n",
       "      <td>basic.4y</td>\n",
       "      <td>unknown</td>\n",
       "      <td>yes</td>\n",
       "      <td>no</td>\n",
       "      <td>cellular</td>\n",
       "      <td>aug</td>\n",
       "      <td>thu</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>999</td>\n",
       "      <td>0</td>\n",
       "      <td>nonexistent</td>\n",
       "      <td>1.4</td>\n",
       "      <td>93.444</td>\n",
       "      <td>-36.1</td>\n",
       "      <td>4.963</td>\n",
       "      <td>5228.1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>53</td>\n",
       "      <td>technician</td>\n",
       "      <td>married</td>\n",
       "      <td>unknown</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>cellular</td>\n",
       "      <td>nov</td>\n",
       "      <td>fri</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>999</td>\n",
       "      <td>0</td>\n",
       "      <td>nonexistent</td>\n",
       "      <td>-0.1</td>\n",
       "      <td>93.200</td>\n",
       "      <td>-42.0</td>\n",
       "      <td>4.021</td>\n",
       "      <td>5195.8</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>28</td>\n",
       "      <td>management</td>\n",
       "      <td>single</td>\n",
       "      <td>university.degree</td>\n",
       "      <td>no</td>\n",
       "      <td>yes</td>\n",
       "      <td>no</td>\n",
       "      <td>cellular</td>\n",
       "      <td>jun</td>\n",
       "      <td>thu</td>\n",
       "      <td>...</td>\n",
       "      <td>3</td>\n",
       "      <td>6</td>\n",
       "      <td>2</td>\n",
       "      <td>success</td>\n",
       "      <td>-1.7</td>\n",
       "      <td>94.055</td>\n",
       "      <td>-39.8</td>\n",
       "      <td>0.729</td>\n",
       "      <td>4991.6</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>39</td>\n",
       "      <td>services</td>\n",
       "      <td>married</td>\n",
       "      <td>high.school</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>cellular</td>\n",
       "      <td>apr</td>\n",
       "      <td>fri</td>\n",
       "      <td>...</td>\n",
       "      <td>2</td>\n",
       "      <td>999</td>\n",
       "      <td>0</td>\n",
       "      <td>nonexistent</td>\n",
       "      <td>-1.8</td>\n",
       "      <td>93.075</td>\n",
       "      <td>-47.1</td>\n",
       "      <td>1.405</td>\n",
       "      <td>5099.1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>55</td>\n",
       "      <td>retired</td>\n",
       "      <td>married</td>\n",
       "      <td>basic.4y</td>\n",
       "      <td>no</td>\n",
       "      <td>yes</td>\n",
       "      <td>no</td>\n",
       "      <td>cellular</td>\n",
       "      <td>aug</td>\n",
       "      <td>fri</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>success</td>\n",
       "      <td>-2.9</td>\n",
       "      <td>92.201</td>\n",
       "      <td>-31.4</td>\n",
       "      <td>0.869</td>\n",
       "      <td>5076.2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 21 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   age          job  marital          education  default housing loan  \\\n",
       "0   44  blue-collar  married           basic.4y  unknown     yes   no   \n",
       "1   53   technician  married            unknown       no      no   no   \n",
       "2   28   management   single  university.degree       no     yes   no   \n",
       "3   39     services  married        high.school       no      no   no   \n",
       "4   55      retired  married           basic.4y       no     yes   no   \n",
       "\n",
       "    contact month day_of_week ...  campaign  pdays  previous     poutcome  \\\n",
       "0  cellular   aug         thu ...         1    999         0  nonexistent   \n",
       "1  cellular   nov         fri ...         1    999         0  nonexistent   \n",
       "2  cellular   jun         thu ...         3      6         2      success   \n",
       "3  cellular   apr         fri ...         2    999         0  nonexistent   \n",
       "4  cellular   aug         fri ...         1      3         1      success   \n",
       "\n",
       "  emp_var_rate  cons_price_idx  cons_conf_idx  euribor3m  nr_employed  y  \n",
       "0          1.4          93.444          -36.1      4.963       5228.1  0  \n",
       "1         -0.1          93.200          -42.0      4.021       5195.8  0  \n",
       "2         -1.7          94.055          -39.8      0.729       4991.6  1  \n",
       "3         -1.8          93.075          -47.1      1.405       5099.1  0  \n",
       "4         -2.9          92.201          -31.4      0.869       5076.2  1  \n",
       "\n",
       "[5 rows x 21 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Input variables"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "1 - age (numeric)\n",
    "\n",
    "2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')\n",
    "\n",
    "3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)\n",
    "\n",
    "4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')\n",
    "\n",
    "5 - default: has credit in default? (categorical: 'no','yes','unknown')\n",
    "\n",
    "6 - housing: has housing loan? (categorical: 'no','yes','unknown')\n",
    "\n",
    "7 - loan: has personal loan? (categorical: 'no','yes','unknown')\n",
    "\n",
    "8 - contact: contact communication type (categorical: 'cellular','telephone')\n",
    "\n",
    "9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')\n",
    "\n",
    "10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')\n",
    "\n",
    "11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.\n",
    "\n",
    "12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)\n",
    "\n",
    "13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)\n",
    "\n",
    "14 - previous: number of contacts performed before this campaign and for this client (numeric)\n",
    "\n",
    "15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')\n",
    "\n",
    "16 - emp.var.rate: employment variation rate - (numeric)\n",
    "\n",
    "17 - cons.price.idx: consumer price index - (numeric)\n",
    "\n",
    "18 - cons.conf.idx: consumer confidence index - (numeric) \n",
    "\n",
    "19 - euribor3m: euribor 3 month rate - (numeric)\n",
    "\n",
    "20 - nr.employed: number of employees - (numeric)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Predict variable (desired target):\n",
    "\n",
    "y - has the client subscribed a term deposit? (binary: '1','0')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The education column of the dataset has many categories and we need to reduce the categories for a better modelling. The education column has the following categories:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['basic.4y', 'unknown', 'university.degree', 'high.school',\n",
       "       'basic.9y', 'professional.course', 'basic.6y', 'illiterate'], dtype=object)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['education'].unique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let us group \"basic.4y\", \"basic.9y\" and \"basic.6y\" together and call them \"basic\"."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])\n",
    "data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])\n",
    "data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "After grouping, this is the columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Basic', 'unknown', 'university.degree', 'high.school',\n",
       "       'professional.course', 'illiterate'], dtype=object)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['education'].unique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data exploration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    36548\n",
       "1     4640\n",
       "Name: y, dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['y'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1b342ceefd0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='y',data=data, palette='hls')\n",
    "plt.show()\n",
    "plt.savefig('count_plot')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There are 36548 no's and 4640 yes's in the outcome variables."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's get a sense of the numbers across the two classes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "      <th>emp_var_rate</th>\n",
       "      <th>cons_price_idx</th>\n",
       "      <th>cons_conf_idx</th>\n",
       "      <th>euribor3m</th>\n",
       "      <th>nr_employed</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>y</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>39.911185</td>\n",
       "      <td>220.844807</td>\n",
       "      <td>2.633085</td>\n",
       "      <td>984.113878</td>\n",
       "      <td>0.132374</td>\n",
       "      <td>0.248875</td>\n",
       "      <td>93.603757</td>\n",
       "      <td>-40.593097</td>\n",
       "      <td>3.811491</td>\n",
       "      <td>5176.166600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>40.913147</td>\n",
       "      <td>553.191164</td>\n",
       "      <td>2.051724</td>\n",
       "      <td>792.035560</td>\n",
       "      <td>0.492672</td>\n",
       "      <td>-1.233448</td>\n",
       "      <td>93.354386</td>\n",
       "      <td>-39.789784</td>\n",
       "      <td>2.123135</td>\n",
       "      <td>5095.115991</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         age    duration  campaign       pdays  previous  emp_var_rate  \\\n",
       "y                                                                        \n",
       "0  39.911185  220.844807  2.633085  984.113878  0.132374      0.248875   \n",
       "1  40.913147  553.191164  2.051724  792.035560  0.492672     -1.233448   \n",
       "\n",
       "   cons_price_idx  cons_conf_idx  euribor3m  nr_employed  \n",
       "y                                                         \n",
       "0       93.603757     -40.593097   3.811491  5176.166600  \n",
       "1       93.354386     -39.789784   2.123135  5095.115991  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.groupby('y').mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Observations:\n",
    "\n",
    "The average age of customers who bought the term deposit is higher than that of the customers who didn't.\n",
    "The pdays (days since the customer was last contacted) is understandably lower for the customers who bought it. The lower the pdays, the better the memory of the last call and hence the better chances of a sale.\n",
    "Surprisingly, campaigns (number of contacts or calls made during the current campaign) are lower for customers who bought the term deposit."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can calculate categorical means for other categorical variables such as education and marital status to get a more detailed sense of our data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "      <th>emp_var_rate</th>\n",
       "      <th>cons_price_idx</th>\n",
       "      <th>cons_conf_idx</th>\n",
       "      <th>euribor3m</th>\n",
       "      <th>nr_employed</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>job</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>admin.</th>\n",
       "      <td>38.187296</td>\n",
       "      <td>254.312128</td>\n",
       "      <td>2.623489</td>\n",
       "      <td>954.319229</td>\n",
       "      <td>0.189023</td>\n",
       "      <td>0.015563</td>\n",
       "      <td>93.534054</td>\n",
       "      <td>-40.245433</td>\n",
       "      <td>3.550274</td>\n",
       "      <td>5164.125350</td>\n",
       "      <td>0.129726</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>blue-collar</th>\n",
       "      <td>39.555760</td>\n",
       "      <td>264.542360</td>\n",
       "      <td>2.558461</td>\n",
       "      <td>985.160363</td>\n",
       "      <td>0.122542</td>\n",
       "      <td>0.248995</td>\n",
       "      <td>93.656656</td>\n",
       "      <td>-41.375816</td>\n",
       "      <td>3.771996</td>\n",
       "      <td>5175.615150</td>\n",
       "      <td>0.068943</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>entrepreneur</th>\n",
       "      <td>41.723214</td>\n",
       "      <td>263.267857</td>\n",
       "      <td>2.535714</td>\n",
       "      <td>981.267170</td>\n",
       "      <td>0.138736</td>\n",
       "      <td>0.158723</td>\n",
       "      <td>93.605372</td>\n",
       "      <td>-41.283654</td>\n",
       "      <td>3.791120</td>\n",
       "      <td>5176.313530</td>\n",
       "      <td>0.085165</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>housemaid</th>\n",
       "      <td>45.500000</td>\n",
       "      <td>250.454717</td>\n",
       "      <td>2.639623</td>\n",
       "      <td>960.579245</td>\n",
       "      <td>0.137736</td>\n",
       "      <td>0.433396</td>\n",
       "      <td>93.676576</td>\n",
       "      <td>-39.495283</td>\n",
       "      <td>4.009645</td>\n",
       "      <td>5179.529623</td>\n",
       "      <td>0.100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>management</th>\n",
       "      <td>42.362859</td>\n",
       "      <td>257.058140</td>\n",
       "      <td>2.476060</td>\n",
       "      <td>962.647059</td>\n",
       "      <td>0.185021</td>\n",
       "      <td>-0.012688</td>\n",
       "      <td>93.522755</td>\n",
       "      <td>-40.489466</td>\n",
       "      <td>3.611316</td>\n",
       "      <td>5166.650513</td>\n",
       "      <td>0.112175</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>retired</th>\n",
       "      <td>62.027326</td>\n",
       "      <td>273.712209</td>\n",
       "      <td>2.476744</td>\n",
       "      <td>897.936047</td>\n",
       "      <td>0.327326</td>\n",
       "      <td>-0.698314</td>\n",
       "      <td>93.430786</td>\n",
       "      <td>-38.573081</td>\n",
       "      <td>2.770066</td>\n",
       "      <td>5122.262151</td>\n",
       "      <td>0.252326</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>self-employed</th>\n",
       "      <td>39.949331</td>\n",
       "      <td>264.142153</td>\n",
       "      <td>2.660802</td>\n",
       "      <td>976.621393</td>\n",
       "      <td>0.143561</td>\n",
       "      <td>0.094159</td>\n",
       "      <td>93.559982</td>\n",
       "      <td>-40.488107</td>\n",
       "      <td>3.689376</td>\n",
       "      <td>5170.674384</td>\n",
       "      <td>0.104856</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>services</th>\n",
       "      <td>37.926430</td>\n",
       "      <td>258.398085</td>\n",
       "      <td>2.587805</td>\n",
       "      <td>979.974049</td>\n",
       "      <td>0.154951</td>\n",
       "      <td>0.175359</td>\n",
       "      <td>93.634659</td>\n",
       "      <td>-41.290048</td>\n",
       "      <td>3.699187</td>\n",
       "      <td>5171.600126</td>\n",
       "      <td>0.081381</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>student</th>\n",
       "      <td>25.894857</td>\n",
       "      <td>283.683429</td>\n",
       "      <td>2.104000</td>\n",
       "      <td>840.217143</td>\n",
       "      <td>0.524571</td>\n",
       "      <td>-1.408000</td>\n",
       "      <td>93.331613</td>\n",
       "      <td>-40.187543</td>\n",
       "      <td>1.884224</td>\n",
       "      <td>5085.939086</td>\n",
       "      <td>0.314286</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>technician</th>\n",
       "      <td>38.507638</td>\n",
       "      <td>250.232241</td>\n",
       "      <td>2.577339</td>\n",
       "      <td>964.408127</td>\n",
       "      <td>0.153789</td>\n",
       "      <td>0.274566</td>\n",
       "      <td>93.561471</td>\n",
       "      <td>-39.927569</td>\n",
       "      <td>3.820401</td>\n",
       "      <td>5175.648391</td>\n",
       "      <td>0.108260</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unemployed</th>\n",
       "      <td>39.733728</td>\n",
       "      <td>249.451677</td>\n",
       "      <td>2.564103</td>\n",
       "      <td>935.316568</td>\n",
       "      <td>0.199211</td>\n",
       "      <td>-0.111736</td>\n",
       "      <td>93.563781</td>\n",
       "      <td>-40.007594</td>\n",
       "      <td>3.466583</td>\n",
       "      <td>5157.156509</td>\n",
       "      <td>0.142012</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unknown</th>\n",
       "      <td>45.563636</td>\n",
       "      <td>239.675758</td>\n",
       "      <td>2.648485</td>\n",
       "      <td>938.727273</td>\n",
       "      <td>0.154545</td>\n",
       "      <td>0.357879</td>\n",
       "      <td>93.718942</td>\n",
       "      <td>-38.797879</td>\n",
       "      <td>3.949033</td>\n",
       "      <td>5172.931818</td>\n",
       "      <td>0.112121</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                     age    duration  campaign       pdays  previous  \\\n",
       "job                                                                    \n",
       "admin.         38.187296  254.312128  2.623489  954.319229  0.189023   \n",
       "blue-collar    39.555760  264.542360  2.558461  985.160363  0.122542   \n",
       "entrepreneur   41.723214  263.267857  2.535714  981.267170  0.138736   \n",
       "housemaid      45.500000  250.454717  2.639623  960.579245  0.137736   \n",
       "management     42.362859  257.058140  2.476060  962.647059  0.185021   \n",
       "retired        62.027326  273.712209  2.476744  897.936047  0.327326   \n",
       "self-employed  39.949331  264.142153  2.660802  976.621393  0.143561   \n",
       "services       37.926430  258.398085  2.587805  979.974049  0.154951   \n",
       "student        25.894857  283.683429  2.104000  840.217143  0.524571   \n",
       "technician     38.507638  250.232241  2.577339  964.408127  0.153789   \n",
       "unemployed     39.733728  249.451677  2.564103  935.316568  0.199211   \n",
       "unknown        45.563636  239.675758  2.648485  938.727273  0.154545   \n",
       "\n",
       "               emp_var_rate  cons_price_idx  cons_conf_idx  euribor3m  \\\n",
       "job                                                                     \n",
       "admin.             0.015563       93.534054     -40.245433   3.550274   \n",
       "blue-collar        0.248995       93.656656     -41.375816   3.771996   \n",
       "entrepreneur       0.158723       93.605372     -41.283654   3.791120   \n",
       "housemaid          0.433396       93.676576     -39.495283   4.009645   \n",
       "management        -0.012688       93.522755     -40.489466   3.611316   \n",
       "retired           -0.698314       93.430786     -38.573081   2.770066   \n",
       "self-employed      0.094159       93.559982     -40.488107   3.689376   \n",
       "services           0.175359       93.634659     -41.290048   3.699187   \n",
       "student           -1.408000       93.331613     -40.187543   1.884224   \n",
       "technician         0.274566       93.561471     -39.927569   3.820401   \n",
       "unemployed        -0.111736       93.563781     -40.007594   3.466583   \n",
       "unknown            0.357879       93.718942     -38.797879   3.949033   \n",
       "\n",
       "               nr_employed         y  \n",
       "job                                   \n",
       "admin.         5164.125350  0.129726  \n",
       "blue-collar    5175.615150  0.068943  \n",
       "entrepreneur   5176.313530  0.085165  \n",
       "housemaid      5179.529623  0.100000  \n",
       "management     5166.650513  0.112175  \n",
       "retired        5122.262151  0.252326  \n",
       "self-employed  5170.674384  0.104856  \n",
       "services       5171.600126  0.081381  \n",
       "student        5085.939086  0.314286  \n",
       "technician     5175.648391  0.108260  \n",
       "unemployed     5157.156509  0.142012  \n",
       "unknown        5172.931818  0.112121  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.groupby('job').mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "      <th>emp_var_rate</th>\n",
       "      <th>cons_price_idx</th>\n",
       "      <th>cons_conf_idx</th>\n",
       "      <th>euribor3m</th>\n",
       "      <th>nr_employed</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>marital</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>divorced</th>\n",
       "      <td>44.899393</td>\n",
       "      <td>253.790330</td>\n",
       "      <td>2.61340</td>\n",
       "      <td>968.639853</td>\n",
       "      <td>0.168690</td>\n",
       "      <td>0.163985</td>\n",
       "      <td>93.606563</td>\n",
       "      <td>-40.707069</td>\n",
       "      <td>3.715603</td>\n",
       "      <td>5170.878643</td>\n",
       "      <td>0.103209</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>married</th>\n",
       "      <td>42.307165</td>\n",
       "      <td>257.438623</td>\n",
       "      <td>2.57281</td>\n",
       "      <td>967.247673</td>\n",
       "      <td>0.155608</td>\n",
       "      <td>0.183625</td>\n",
       "      <td>93.597367</td>\n",
       "      <td>-40.270659</td>\n",
       "      <td>3.745832</td>\n",
       "      <td>5171.848772</td>\n",
       "      <td>0.101573</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>single</th>\n",
       "      <td>33.158714</td>\n",
       "      <td>261.524378</td>\n",
       "      <td>2.53380</td>\n",
       "      <td>949.909578</td>\n",
       "      <td>0.211359</td>\n",
       "      <td>-0.167989</td>\n",
       "      <td>93.517300</td>\n",
       "      <td>-40.918698</td>\n",
       "      <td>3.317447</td>\n",
       "      <td>5155.199265</td>\n",
       "      <td>0.140041</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unknown</th>\n",
       "      <td>40.275000</td>\n",
       "      <td>312.725000</td>\n",
       "      <td>3.18750</td>\n",
       "      <td>937.100000</td>\n",
       "      <td>0.275000</td>\n",
       "      <td>-0.221250</td>\n",
       "      <td>93.471250</td>\n",
       "      <td>-40.820000</td>\n",
       "      <td>3.313038</td>\n",
       "      <td>5157.393750</td>\n",
       "      <td>0.150000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                age    duration  campaign       pdays  previous  emp_var_rate  \\\n",
       "marital                                                                         \n",
       "divorced  44.899393  253.790330   2.61340  968.639853  0.168690      0.163985   \n",
       "married   42.307165  257.438623   2.57281  967.247673  0.155608      0.183625   \n",
       "single    33.158714  261.524378   2.53380  949.909578  0.211359     -0.167989   \n",
       "unknown   40.275000  312.725000   3.18750  937.100000  0.275000     -0.221250   \n",
       "\n",
       "          cons_price_idx  cons_conf_idx  euribor3m  nr_employed         y  \n",
       "marital                                                                    \n",
       "divorced       93.606563     -40.707069   3.715603  5170.878643  0.103209  \n",
       "married        93.597367     -40.270659   3.745832  5171.848772  0.101573  \n",
       "single         93.517300     -40.918698   3.317447  5155.199265  0.140041  \n",
       "unknown        93.471250     -40.820000   3.313038  5157.393750  0.150000  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.groupby('marital').mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "      <th>emp_var_rate</th>\n",
       "      <th>cons_price_idx</th>\n",
       "      <th>cons_conf_idx</th>\n",
       "      <th>euribor3m</th>\n",
       "      <th>nr_employed</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>education</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Basic</th>\n",
       "      <td>42.163910</td>\n",
       "      <td>263.043874</td>\n",
       "      <td>2.559498</td>\n",
       "      <td>974.877967</td>\n",
       "      <td>0.141053</td>\n",
       "      <td>0.191329</td>\n",
       "      <td>93.639933</td>\n",
       "      <td>-40.927595</td>\n",
       "      <td>3.729654</td>\n",
       "      <td>5172.014113</td>\n",
       "      <td>0.087029</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>high.school</th>\n",
       "      <td>37.998213</td>\n",
       "      <td>260.886810</td>\n",
       "      <td>2.568576</td>\n",
       "      <td>964.358382</td>\n",
       "      <td>0.185917</td>\n",
       "      <td>0.032937</td>\n",
       "      <td>93.584857</td>\n",
       "      <td>-40.940641</td>\n",
       "      <td>3.556157</td>\n",
       "      <td>5164.994735</td>\n",
       "      <td>0.108355</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>illiterate</th>\n",
       "      <td>48.500000</td>\n",
       "      <td>276.777778</td>\n",
       "      <td>2.277778</td>\n",
       "      <td>943.833333</td>\n",
       "      <td>0.111111</td>\n",
       "      <td>-0.133333</td>\n",
       "      <td>93.317333</td>\n",
       "      <td>-39.950000</td>\n",
       "      <td>3.516556</td>\n",
       "      <td>5171.777778</td>\n",
       "      <td>0.222222</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>professional.course</th>\n",
       "      <td>40.080107</td>\n",
       "      <td>252.533855</td>\n",
       "      <td>2.586115</td>\n",
       "      <td>960.765974</td>\n",
       "      <td>0.163075</td>\n",
       "      <td>0.173012</td>\n",
       "      <td>93.569864</td>\n",
       "      <td>-40.124108</td>\n",
       "      <td>3.710457</td>\n",
       "      <td>5170.155979</td>\n",
       "      <td>0.113485</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>university.degree</th>\n",
       "      <td>38.879191</td>\n",
       "      <td>253.223373</td>\n",
       "      <td>2.563527</td>\n",
       "      <td>951.807692</td>\n",
       "      <td>0.192390</td>\n",
       "      <td>-0.028090</td>\n",
       "      <td>93.493466</td>\n",
       "      <td>-39.975805</td>\n",
       "      <td>3.529663</td>\n",
       "      <td>5163.226298</td>\n",
       "      <td>0.137245</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unknown</th>\n",
       "      <td>43.481225</td>\n",
       "      <td>262.390526</td>\n",
       "      <td>2.596187</td>\n",
       "      <td>942.830734</td>\n",
       "      <td>0.226459</td>\n",
       "      <td>0.059099</td>\n",
       "      <td>93.658615</td>\n",
       "      <td>-39.877816</td>\n",
       "      <td>3.571098</td>\n",
       "      <td>5159.549509</td>\n",
       "      <td>0.145003</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                           age    duration  campaign       pdays  previous  \\\n",
       "education                                                                    \n",
       "Basic                42.163910  263.043874  2.559498  974.877967  0.141053   \n",
       "high.school          37.998213  260.886810  2.568576  964.358382  0.185917   \n",
       "illiterate           48.500000  276.777778  2.277778  943.833333  0.111111   \n",
       "professional.course  40.080107  252.533855  2.586115  960.765974  0.163075   \n",
       "university.degree    38.879191  253.223373  2.563527  951.807692  0.192390   \n",
       "unknown              43.481225  262.390526  2.596187  942.830734  0.226459   \n",
       "\n",
       "                     emp_var_rate  cons_price_idx  cons_conf_idx  euribor3m  \\\n",
       "education                                                                     \n",
       "Basic                    0.191329       93.639933     -40.927595   3.729654   \n",
       "high.school              0.032937       93.584857     -40.940641   3.556157   \n",
       "illiterate              -0.133333       93.317333     -39.950000   3.516556   \n",
       "professional.course      0.173012       93.569864     -40.124108   3.710457   \n",
       "university.degree       -0.028090       93.493466     -39.975805   3.529663   \n",
       "unknown                  0.059099       93.658615     -39.877816   3.571098   \n",
       "\n",
       "                     nr_employed         y  \n",
       "education                                   \n",
       "Basic                5172.014113  0.087029  \n",
       "high.school          5164.994735  0.108355  \n",
       "illiterate           5171.777778  0.222222  \n",
       "professional.course  5170.155979  0.113485  \n",
       "university.degree    5163.226298  0.137245  \n",
       "unknown              5159.549509  0.145003  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.groupby('education').mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Visualizations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1b347d13dd8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "%matplotlib inline\n",
    "pd.crosstab(data.job,data.y).plot(kind='bar')\n",
    "plt.title('Purchase Frequency for Job Title')\n",
    "plt.xlabel('Job')\n",
    "plt.ylabel('Frequency of Purchase')\n",
    "plt.savefig('purchase_fre_job')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The frequency of purchase of the deposit depends a great deal on the job title. Thus, the job title can be a good predictor of the outcome variable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1b346c65c50>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "table=pd.crosstab(data.marital,data.y)\n",
    "table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)\n",
    "plt.title('Stacked Bar Chart of Marital Status vs Purchase')\n",
    "plt.xlabel('Marital Status')\n",
    "plt.ylabel('Proportion of Customers')\n",
    "plt.savefig('mariral_vs_pur_stack')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Hard to see, but the marital status does not seem a strong predictor for the outcome variable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\nAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1b3486a5080>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "table=pd.crosstab(data.education,data.y)\n",
    "table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)\n",
    "plt.title('Stacked Bar Chart of Education vs Purchase')\n",
    "plt.xlabel('Education')\n",
    "plt.ylabel('Proportion of Customers')\n",
    "plt.savefig('edu_vs_pur_stack')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Education seems a good predictor of the outcome variable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1b348769898>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "pd.crosstab(data.day_of_week,data.y).plot(kind='bar')\n",
    "plt.title('Purchase Frequency for Day of Week')\n",
    "plt.xlabel('Day of Week')\n",
    "plt.ylabel('Frequency of Purchase')\n",
    "plt.savefig('pur_dayofweek_bar')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Day of week may not be a good predictor of the outcome"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x16312119080>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "pd.crosstab(data.month,data.y).plot(kind='bar')\n",
    "plt.title('Purchase Frequency for Month')\n",
    "plt.xlabel('Month')\n",
    "plt.ylabel('Frequency of Purchase')\n",
    "plt.savefig('pur_fre_month_bar')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Month might be a good predictor of the outcome variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1b347e47f60>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "data.age.hist()\n",
    "plt.title('Histogram of Age')\n",
    "plt.xlabel('Age')\n",
    "plt.ylabel('Frequency')\n",
    "plt.savefig('hist_age')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The most of the customers of the bank in this dataset are in the age range of 30-40."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1b347e7eeb8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "pd.crosstab(data.poutcome,data.y).plot(kind='bar')\n",
    "plt.title('Purchase Frequency for Poutcome')\n",
    "plt.xlabel('Poutcome')\n",
    "plt.ylabel('Frequency of Purchase')\n",
    "plt.savefig('pur_fre_pout_bar')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Poutcome seems to be a good predictor of the outcome variable."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Create dummy variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']\n",
    "for var in cat_vars:\n",
    "    cat_list='var'+'_'+var\n",
    "    cat_list = pd.get_dummies(data[var], prefix=var)\n",
    "    data1=data.join(cat_list)\n",
    "    data=data1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']\n",
    "data_vars=data.columns.values.tolist()\n",
    "to_keep=[i for i in data_vars if i not in cat_vars]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate',\n",
       "       'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed', 'y',\n",
       "       'job_admin.', 'job_blue-collar', 'job_entrepreneur',\n",
       "       'job_housemaid', 'job_management', 'job_retired',\n",
       "       'job_self-employed', 'job_services', 'job_student',\n",
       "       'job_technician', 'job_unemployed', 'job_unknown',\n",
       "       'marital_divorced', 'marital_married', 'marital_single',\n",
       "       'marital_unknown', 'education_Basic', 'education_high.school',\n",
       "       'education_illiterate', 'education_professional.course',\n",
       "       'education_university.degree', 'education_unknown', 'default_no',\n",
       "       'default_unknown', 'default_yes', 'housing_no', 'housing_unknown',\n",
       "       'housing_yes', 'loan_no', 'loan_unknown', 'loan_yes',\n",
       "       'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug',\n",
       "       'month_dec', 'month_jul', 'month_jun', 'month_mar', 'month_may',\n",
       "       'month_nov', 'month_oct', 'month_sep', 'day_of_week_fri',\n",
       "       'day_of_week_mon', 'day_of_week_thu', 'day_of_week_tue',\n",
       "       'day_of_week_wed', 'poutcome_failure', 'poutcome_nonexistent',\n",
       "       'poutcome_success'], dtype=object)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_final=data[to_keep]\n",
    "data_final.columns.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "data_final_vars=data_final.columns.values.tolist()\n",
    "y=['y']\n",
    "X=[i for i in data_final_vars if i not in y]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Feature Selection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Susan\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:526: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[False False False False  True False False False  True False False  True\n",
      " False False False  True False  True  True False False False False False\n",
      " False False False False False False False False  True False False False\n",
      " False False False False False False  True  True  True False False False\n",
      "  True  True  True False False False  True False False  True  True  True\n",
      "  True]\n",
      "[35 33 12 40  1 13 17 16  1 27 11  1 24 39 42  1 31  1  1 19 21 41  2  3  4\n",
      " 43  6  7 38  8 10 15  1 14 44 36 29 37 20 30 28 23  1  1  1 18 22 25  1  1\n",
      "  1 32  5  9  1 34 26  1  1  1  1]\n"
     ]
    }
   ],
   "source": [
    "from sklearn import datasets\n",
    "from sklearn.feature_selection import RFE\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "logreg = LogisticRegression()\n",
    "\n",
    "rfe = RFE(logreg, 18)\n",
    "rfe = rfe.fit(data_final[X], data_final[y] )\n",
    "print(rfe.support_)\n",
    "print(rfe.ranking_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The Recursive Feature Elimination (RFE) has helped us select the following features: \"previous\", \"euribor3m\", \"job_blue-collar\", \"job_retired\", \"job_services\", \"job_student\", \"default_no\", \"month_aug\", \"month_dec\", \"month_jul\", \"month_nov\", \"month_oct\", \"month_sep\", \"day_of_week_fri\", \"day_of_week_wed\", \"poutcome_failure\", \"poutcome_nonexistent\", \"poutcome_success\"."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "cols=[\"previous\", \"euribor3m\", \"job_blue-collar\", \"job_retired\", \"job_services\", \"job_student\", \"default_no\", \n",
    "      \"month_aug\", \"month_dec\", \"month_jul\", \"month_nov\", \"month_oct\", \"month_sep\", \"day_of_week_fri\", \"day_of_week_wed\", \n",
    "      \"poutcome_failure\", \"poutcome_nonexistent\", \"poutcome_success\"] \n",
    "X=data_final[cols]\n",
    "y=data_final['y']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Implementing the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Optimization terminated successfully.\n",
      "         Current function value: 0.287116\n",
      "         Iterations 7\n",
      "                           Logit Regression Results                           \n",
      "==============================================================================\n",
      "Dep. Variable:                      y   No. Observations:                41188\n",
      "Model:                          Logit   Df Residuals:                    41170\n",
      "Method:                           MLE   Df Model:                           17\n",
      "Date:                Sat, 18 Nov 2017   Pseudo R-squ.:                  0.1844\n",
      "Time:                        02:47:55   Log-Likelihood:                -11826.\n",
      "converged:                       True   LL-Null:                       -14499.\n",
      "                                        LLR p-value:                     0.000\n",
      "========================================================================================\n",
      "                           coef    std err          z      P>|z|      [0.025      0.975]\n",
      "----------------------------------------------------------------------------------------\n",
      "previous                 0.2385      0.051      4.642      0.000       0.138       0.339\n",
      "euribor3m               -0.4981      0.012    -40.386      0.000      -0.522      -0.474\n",
      "job_blue-collar         -0.3222      0.049     -6.549      0.000      -0.419      -0.226\n",
      "job_retired              0.3821      0.069      5.552      0.000       0.247       0.517\n",
      "job_services            -0.2423      0.065     -3.701      0.000      -0.371      -0.114\n",
      "job_student              0.3540      0.086      4.107      0.000       0.185       0.523\n",
      "default_no               0.3312      0.056      5.943      0.000       0.222       0.440\n",
      "month_aug                0.4272      0.055      7.770      0.000       0.319       0.535\n",
      "month_dec                0.8061      0.163      4.948      0.000       0.487       1.125\n",
      "month_jul                0.7319      0.056     13.094      0.000       0.622       0.841\n",
      "month_nov                0.2706      0.064      4.249      0.000       0.146       0.395\n",
      "month_oct                0.8043      0.087      9.258      0.000       0.634       0.975\n",
      "month_sep                0.5906      0.096      6.160      0.000       0.403       0.778\n",
      "day_of_week_fri         -0.0044      0.046     -0.097      0.923      -0.094       0.085\n",
      "day_of_week_wed          0.1226      0.044      2.771      0.006       0.036       0.209\n",
      "poutcome_failure        -1.8438      0.100    -18.412      0.000      -2.040      -1.647\n",
      "poutcome_nonexistent    -1.1344      0.070    -16.253      0.000      -1.271      -0.998\n",
      "poutcome_success         0.0912      0.114      0.803      0.422      -0.131       0.314\n",
      "========================================================================================\n"
     ]
    }
   ],
   "source": [
    "import statsmodels.api as sm\n",
    "logit_model=sm.Logit(y,X)\n",
    "result=logit_model.fit()\n",
    "print(result.summary())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The p-values for most of the variables are very small, therefore, most of them are significant to the model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Logistic Regression Model Fitting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n",
       "          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,\n",
       "          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,\n",
       "          verbose=0, warm_start=False)"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn import metrics\n",
    "logreg = LogisticRegression()\n",
    "logreg.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Predicting the test set results and caculating the accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "y_pred = logreg.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy of logistic regression classifier on test set: 0.90\n"
     ]
    }
   ],
   "source": [
    "print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Cross Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "10-fold cross validation average accuracy: 0.897\n"
     ]
    }
   ],
   "source": [
    "from sklearn import model_selection\n",
    "from sklearn.model_selection import cross_val_score\n",
    "kfold = model_selection.KFold(n_splits=10, random_state=7)\n",
    "modelCV = LogisticRegression()\n",
    "scoring = 'accuracy'\n",
    "results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)\n",
    "print(\"10-fold cross validation average accuracy: %.3f\" % (results.mean()))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[10872   109]\n",
      " [ 1122   254]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "confusion_matrix = confusion_matrix(y_test, y_pred)\n",
    "print(confusion_matrix)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The result is telling us that we have 10872+254 correct predictions and 1122+109 incorrect predictions."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy of logistic regression classifier on test set: 0.90\n"
     ]
    }
   ],
   "source": [
    "print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Compute precision, recall, F-measure and support\n",
    "\n",
    "The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.\n",
    "\n",
    "The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.\n",
    "\n",
    "The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.\n",
    "\n",
    "The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision are equally important.\n",
    "\n",
    "The support is the number of occurrences of each class in y_test."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             precision    recall  f1-score   support\n",
      "\n",
      "          0       0.91      0.99      0.95     10981\n",
      "          1       0.70      0.18      0.29      1376\n",
      "\n",
      "avg / total       0.88      0.90      0.87     12357\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Interpretation: \n",
    "\n",
    "Of the entire test set, 88% of the promoted term deposit were the term deposit that the customers liked. Of the entire test set, 90% of the customer's preferred term deposit were promoted."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### ROC Curvefrom sklearn import metrics\n",
    "from ggplot import *\n",
    "\n",
    "prob = clf1.predict_proba(X_test)[:,1]\n",
    "fpr, sensitivity, _ = metrics.roc_curve(Y_test, prob)\n",
    "\n",
    "df = pd.DataFrame(dict(fpr=fpr, sensitivity=sensitivity))\n",
    "ggplot(df, aes(x='fpr', y='sensitivity')) +\\\n",
    "    geom_line() +\\\n",
    "    geom_abline(linetype='dashed')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1b34a82bd68>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.metrics import roc_auc_score\n",
    "from sklearn.metrics import roc_curve\n",
    "logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))\n",
    "fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])\n",
    "plt.figure()\n",
    "plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)\n",
    "plt.plot([0, 1], [0, 1],'r--')\n",
    "plt.xlim([0.0, 1.0])\n",
    "plt.ylim([0.0, 1.05])\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('Receiver operating characteristic')\n",
    "plt.legend(loc=\"lower right\")\n",
    "plt.savefig('Log_ROC')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

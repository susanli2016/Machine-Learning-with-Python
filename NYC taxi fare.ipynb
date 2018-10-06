{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "% matplotlib inline\n",
    "plt.style.use('seaborn-white')\n",
    "pd.set_option('display.float_format', lambda x: '%.3f' % x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data Preprocessing"
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
      "The dataset is 5000000 taxi rides\n"
     ]
    }
   ],
   "source": [
    "taxi = pd.read_csv('ny_taxi.csv', nrows = 5_000_000, parse_dates = ['pickup_datetime']).drop(columns = 'key')\n",
    "print(\"The dataset is {} taxi rides\".format(len(taxi)))"
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
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fare_amount</th>\n",
       "      <th>pickup_datetime</th>\n",
       "      <th>pickup_longitude</th>\n",
       "      <th>pickup_latitude</th>\n",
       "      <th>dropoff_longitude</th>\n",
       "      <th>dropoff_latitude</th>\n",
       "      <th>passenger_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>4.500</td>\n",
       "      <td>2009-06-15 17:26:21</td>\n",
       "      <td>-73.844</td>\n",
       "      <td>40.721</td>\n",
       "      <td>-73.842</td>\n",
       "      <td>40.712</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>16.900</td>\n",
       "      <td>2010-01-05 16:52:16</td>\n",
       "      <td>-74.016</td>\n",
       "      <td>40.711</td>\n",
       "      <td>-73.979</td>\n",
       "      <td>40.782</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5.700</td>\n",
       "      <td>2011-08-18 00:35:00</td>\n",
       "      <td>-73.983</td>\n",
       "      <td>40.761</td>\n",
       "      <td>-73.991</td>\n",
       "      <td>40.751</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7.700</td>\n",
       "      <td>2012-04-21 04:30:42</td>\n",
       "      <td>-73.987</td>\n",
       "      <td>40.733</td>\n",
       "      <td>-73.992</td>\n",
       "      <td>40.758</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.300</td>\n",
       "      <td>2010-03-09 07:51:00</td>\n",
       "      <td>-73.968</td>\n",
       "      <td>40.768</td>\n",
       "      <td>-73.957</td>\n",
       "      <td>40.784</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   fare_amount     pickup_datetime  pickup_longitude  pickup_latitude  \\\n",
       "0        4.500 2009-06-15 17:26:21           -73.844           40.721   \n",
       "1       16.900 2010-01-05 16:52:16           -74.016           40.711   \n",
       "2        5.700 2011-08-18 00:35:00           -73.983           40.761   \n",
       "3        7.700 2012-04-21 04:30:42           -73.987           40.733   \n",
       "4        5.300 2010-03-09 07:51:00           -73.968           40.768   \n",
       "\n",
       "   dropoff_longitude  dropoff_latitude  passenger_count  \n",
       "0            -73.842            40.712                1  \n",
       "1            -73.979            40.782                1  \n",
       "2            -73.991            40.751                2  \n",
       "3            -73.992            40.758                1  \n",
       "4            -73.957            40.784                1  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "taxi.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fare_amount</th>\n",
       "      <th>pickup_longitude</th>\n",
       "      <th>pickup_latitude</th>\n",
       "      <th>dropoff_longitude</th>\n",
       "      <th>dropoff_latitude</th>\n",
       "      <th>passenger_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>5000000.000</td>\n",
       "      <td>5000000.000</td>\n",
       "      <td>5000000.000</td>\n",
       "      <td>4999964.000</td>\n",
       "      <td>4999964.000</td>\n",
       "      <td>5000000.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>11.341</td>\n",
       "      <td>-72.507</td>\n",
       "      <td>39.920</td>\n",
       "      <td>-72.507</td>\n",
       "      <td>39.917</td>\n",
       "      <td>1.685</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>9.820</td>\n",
       "      <td>12.810</td>\n",
       "      <td>8.964</td>\n",
       "      <td>12.848</td>\n",
       "      <td>9.487</td>\n",
       "      <td>1.332</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>-100.000</td>\n",
       "      <td>-3426.609</td>\n",
       "      <td>-3488.080</td>\n",
       "      <td>-3412.653</td>\n",
       "      <td>-3488.080</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>6.000</td>\n",
       "      <td>-73.992</td>\n",
       "      <td>40.735</td>\n",
       "      <td>-73.991</td>\n",
       "      <td>40.734</td>\n",
       "      <td>1.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>8.500</td>\n",
       "      <td>-73.982</td>\n",
       "      <td>40.753</td>\n",
       "      <td>-73.980</td>\n",
       "      <td>40.753</td>\n",
       "      <td>1.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>12.500</td>\n",
       "      <td>-73.967</td>\n",
       "      <td>40.767</td>\n",
       "      <td>-73.964</td>\n",
       "      <td>40.768</td>\n",
       "      <td>2.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1273.310</td>\n",
       "      <td>3439.426</td>\n",
       "      <td>3310.364</td>\n",
       "      <td>3457.622</td>\n",
       "      <td>3345.917</td>\n",
       "      <td>208.000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       fare_amount  pickup_longitude  pickup_latitude  dropoff_longitude  \\\n",
       "count  5000000.000       5000000.000      5000000.000        4999964.000   \n",
       "mean        11.341           -72.507           39.920            -72.507   \n",
       "std          9.820            12.810            8.964             12.848   \n",
       "min       -100.000         -3426.609        -3488.080          -3412.653   \n",
       "25%          6.000           -73.992           40.735            -73.991   \n",
       "50%          8.500           -73.982           40.753            -73.980   \n",
       "75%         12.500           -73.967           40.767            -73.964   \n",
       "max       1273.310          3439.426         3310.364           3457.622   \n",
       "\n",
       "       dropoff_latitude  passenger_count  \n",
       "count       4999964.000      5000000.000  \n",
       "mean             39.917            1.685  \n",
       "std               9.487            1.332  \n",
       "min           -3488.080            0.000  \n",
       "25%              40.734            1.000  \n",
       "50%              40.753            1.000  \n",
       "75%              40.768            2.000  \n",
       "max            3345.917          208.000  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "taxi.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Find something funny? \n",
    "\n",
    "1). Min fare amount is negative.\n",
    "\n",
    "2). Min and Max longitude and latitude look unreal.\n",
    "\n",
    "3). Min passenger count is 0."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We are fixing them now.\n",
    "\n",
    "1). New York city longitudes are around -74 and latitudes are around 41.\n",
    "\n",
    "2). Remove 0 passenger count.\n",
    "\n",
    "3). The initial charge is $2.5, so we are removing fare amount smaller than this amount."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "taxi = taxi[((taxi['pickup_longitude'] > -78) & (taxi['pickup_longitude'] < -70)) & ((taxi['dropoff_longitude'] > -78) & (taxi['dropoff_longitude'] < -70)) & ((taxi['pickup_latitude'] > 37) & (taxi['pickup_latitude'] < 45)) & ((taxi['dropoff_latitude'] > 37) & (taxi['dropoff_latitude'] < 45)) & (taxi['passenger_count'] > 0) & (taxi['fare_amount'] >= 2.5)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fare_amount</th>\n",
       "      <th>pickup_longitude</th>\n",
       "      <th>pickup_latitude</th>\n",
       "      <th>dropoff_longitude</th>\n",
       "      <th>dropoff_latitude</th>\n",
       "      <th>passenger_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>4878498.000</td>\n",
       "      <td>4878498.000</td>\n",
       "      <td>4878498.000</td>\n",
       "      <td>4878498.000</td>\n",
       "      <td>4878498.000</td>\n",
       "      <td>4878498.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>11.333</td>\n",
       "      <td>-73.975</td>\n",
       "      <td>40.751</td>\n",
       "      <td>-73.974</td>\n",
       "      <td>40.751</td>\n",
       "      <td>1.690</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>9.727</td>\n",
       "      <td>0.043</td>\n",
       "      <td>0.034</td>\n",
       "      <td>0.042</td>\n",
       "      <td>0.037</td>\n",
       "      <td>1.314</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>2.500</td>\n",
       "      <td>-77.902</td>\n",
       "      <td>37.031</td>\n",
       "      <td>-77.971</td>\n",
       "      <td>37.031</td>\n",
       "      <td>1.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>6.000</td>\n",
       "      <td>-73.992</td>\n",
       "      <td>40.737</td>\n",
       "      <td>-73.992</td>\n",
       "      <td>40.736</td>\n",
       "      <td>1.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>8.500</td>\n",
       "      <td>-73.982</td>\n",
       "      <td>40.753</td>\n",
       "      <td>-73.981</td>\n",
       "      <td>40.754</td>\n",
       "      <td>1.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>12.500</td>\n",
       "      <td>-73.968</td>\n",
       "      <td>40.768</td>\n",
       "      <td>-73.965</td>\n",
       "      <td>40.768</td>\n",
       "      <td>2.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>952.000</td>\n",
       "      <td>-70.000</td>\n",
       "      <td>44.732</td>\n",
       "      <td>-70.002</td>\n",
       "      <td>44.728</td>\n",
       "      <td>208.000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       fare_amount  pickup_longitude  pickup_latitude  dropoff_longitude  \\\n",
       "count  4878498.000       4878498.000      4878498.000        4878498.000   \n",
       "mean        11.333           -73.975           40.751            -73.974   \n",
       "std          9.727             0.043            0.034              0.042   \n",
       "min          2.500           -77.902           37.031            -77.971   \n",
       "25%          6.000           -73.992           40.737            -73.992   \n",
       "50%          8.500           -73.982           40.753            -73.981   \n",
       "75%         12.500           -73.968           40.768            -73.965   \n",
       "max        952.000           -70.000           44.732            -70.002   \n",
       "\n",
       "       dropoff_latitude  passenger_count  \n",
       "count       4878498.000      4878498.000  \n",
       "mean             40.751            1.690  \n",
       "std               0.037            1.314  \n",
       "min              37.031            1.000  \n",
       "25%              40.736            1.000  \n",
       "50%              40.754            1.000  \n",
       "75%              40.768            2.000  \n",
       "max              44.728          208.000  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "taxi.describe()"
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
       "fare_amount          0\n",
       "pickup_datetime      0\n",
       "pickup_longitude     0\n",
       "pickup_latitude      0\n",
       "dropoff_longitude    0\n",
       "dropoff_latitude     0\n",
       "passenger_count      0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "taxi.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### EDA"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's try to visualize ten taxi rides."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x2184cdbcf98>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "def showrides(df, numlines):\n",
    "  lats = []\n",
    "  lons = []\n",
    "  goodrows = df[df['pickup_longitude'] < -70]\n",
    "  for iter, row in goodrows[:numlines].iterrows():\n",
    "    lons.append(row['pickup_longitude'])\n",
    "    lons.append(row['dropoff_longitude'])\n",
    "    lons.append(None)\n",
    "    lats.append(row['pickup_latitude'])\n",
    "    lats.append(row['dropoff_latitude'])\n",
    "    lats.append(None)\n",
    "\n",
    "  plt.plot(lons, lats)\n",
    "\n",
    "showrides(taxi, 10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Some ride distances are very short, some are pretty long."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Fare amount"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x21859ca0128>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize = (14, 4))\n",
    "n, bins, patches = plt.hist(taxi.fare_amount, 1000, facecolor='blue', alpha=0.75)\n",
    "plt.xlabel('Fare amount')\n",
    "plt.title('Histogram of fare amount')\n",
    "plt.xlim(0, 200)\n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The histogram of fare amount shows that most fare amount are small."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "fare_amount\n",
       "6.500     237317\n",
       "4.500     200433\n",
       "8.500     183203\n",
       "5.700     142304\n",
       "5.300     142293\n",
       "6.100     138084\n",
       "4.900     138049\n",
       "6.900     127239\n",
       "10.500    124046\n",
       "7.300     119606\n",
       "dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "taxi.groupby('fare_amount').size().nlargest(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Interesting, the most common fare amount are very small at only 6.5 and 4.5, they are very short rides."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Passenger count"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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uAc43PYKSUuBqynBtqC2PRI4RdKmwC3AIturG2gbEREt6vT3T75rFCXNBu5YsmQJs2bNGrZdp9NhdNdwdpiMn29ETEx9fX3MmzcPYIc6FNdaerXLKyIiJrgESkRENCKBEhERjUigREREIxIoERHRiARKREQ0IoESERGNSKBEREQjEigREdGIBEpERDQigRIREY1IoERERCMSKBER0YgESkRENCKBEhERjUigREREIxIoERHRiARKREQ0IoESERGNSKBEREQjEigREdGIBEpERDQigRIREY2Y1taKJe0GfNz2XEk7AmcA/cDNwALbayQdBxwArAIW2r6mrbZtvc+IiCha6aFIejfwRWDzOusE4BjbewEdYL6kOcA+wG7Aa4CTWm4bEREtamuX123AQV2PdwEur9MXAS8F9gQW2+63fTcwTdJWLbaNiIgWtRIotr8GPNo1q2O7v06vBGYAWwLLu9oMzG+rbUREtKhXB+XXdE1PB+4HVtTpwfPbahsRES3qVaDcIGlund4fWApcBewnaYqk7YEptu9rsW1ERLSotbO8BjkSOEXSpsAtwCLbqyUtBa6mBNuClttGRESLOv39/etvNcFImg3csWTJEmbNmjVsu06nQzkjeUN1mIyfb0RMTH19fcybNw9gB9t3Dn4+FzZGREQjEigREdGIBEpERDQigRIREY1IoERERCMSKBER0YgESkRENCKBEhERjUigREREIxIoERHRiARKREQ0IoESERGNSKBs5GbOnE2n0xnVz8yZs8f6bUTEJNCr4etjAy1bdhejG/EYli3rNFNMRMQ6pIcSERGNSKBEREQjEigREdGIBEpERDQigRIREY1IoERERCMSKBER0YgESkRENCKBEhERjejplfKSbgCW14d3AJ8HPg2sAhbb/qCkKcC/ATsBDwNvsv0LSS8aTdvevcuIiMmpZ4EiaXMA23O75t0IvAq4HbhQ0hxgNrC57RfXYPgUMB84eTRtbV/fi/cZETFZ9bKHshOwhaTF9XU/AGxm+zYASRcD84BtgG8D2P6+pBdI2rKBtgmUiIgW9fIYyoPA8cB+wFuA0+u8ASuBGcCW/HG3GMDqOm/FKNtGRESLetlD+TnwC9v9wM8lLQf+T9fz04H7gS3q9IAplICYPsq2ERHRol72UA6jHONA0raUMPidpGdI6lB6LkuBq4CX13YvAm6y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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x2185a668e48>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "taxi['passenger_count'].value_counts().plot.bar(color = 'b', edgecolor = 'k');\n",
    "plt.title('Histogram of passenger counts'); plt.xlabel('Passenger counts'); plt.ylabel('Count');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "passenger_count\n",
       "1      3386066\n",
       "2       723885\n",
       "3       214918\n",
       "4       103907\n",
       "5       346169\n",
       "6       103547\n",
       "7            1\n",
       "9            2\n",
       "129          1\n",
       "208          2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "taxi.groupby('passenger_count').size()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Based on the above discovery, we are going to remove taxi rides with passenger_count > 6."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "taxi = taxi.loc[taxi['passenger_count'] <= 6]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "passenger_count\n",
       "1    3386066\n",
       "2     723885\n",
       "3     214918\n",
       "4     103907\n",
       "5     346169\n",
       "6     103547\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "taxi.groupby('passenger_count').size()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fare_amount</th>\n",
       "      <th>pickup_longitude</th>\n",
       "      <th>pickup_latitude</th>\n",
       "      <th>dropoff_longitude</th>\n",
       "      <th>dropoff_latitude</th>\n",
       "      <th>passenger_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>4878498.000</td>\n",
       "      <td>4878498.000</td>\n",
       "      <td>4878498.000</td>\n",
       "      <td>4878498.000</td>\n",
       "      <td>4878498.000</td>\n",
       "      <td>4878498.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>11.333</td>\n",
       "      <td>-73.975</td>\n",
       "      <td>40.751</td>\n",
       "      <td>-73.974</td>\n",
       "      <td>40.751</td>\n",
       "      <td>1.690</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>9.727</td>\n",
       "      <td>0.043</td>\n",
       "      <td>0.034</td>\n",
       "      <td>0.042</td>\n",
       "      <td>0.037</td>\n",
       "      <td>1.314</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>2.500</td>\n",
       "      <td>-77.902</td>\n",
       "      <td>37.031</td>\n",
       "      <td>-77.971</td>\n",
       "      <td>37.031</td>\n",
       "      <td>1.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>6.000</td>\n",
       "      <td>-73.992</td>\n",
       "      <td>40.737</td>\n",
       "      <td>-73.992</td>\n",
       "      <td>40.736</td>\n",
       "      <td>1.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>8.500</td>\n",
       "      <td>-73.982</td>\n",
       "      <td>40.753</td>\n",
       "      <td>-73.981</td>\n",
       "      <td>40.754</td>\n",
       "      <td>1.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>12.500</td>\n",
       "      <td>-73.968</td>\n",
       "      <td>40.768</td>\n",
       "      <td>-73.965</td>\n",
       "      <td>40.768</td>\n",
       "      <td>2.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>952.000</td>\n",
       "      <td>-70.000</td>\n",
       "      <td>44.732</td>\n",
       "      <td>-70.002</td>\n",
       "      <td>44.728</td>\n",
       "      <td>208.000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       fare_amount  pickup_longitude  pickup_latitude  dropoff_longitude  \\\n",
       "count  4878498.000       4878498.000      4878498.000        4878498.000   \n",
       "mean        11.333           -73.975           40.751            -73.974   \n",
       "std          9.727             0.043            0.034              0.042   \n",
       "min          2.500           -77.902           37.031            -77.971   \n",
       "25%          6.000           -73.992           40.737            -73.992   \n",
       "50%          8.500           -73.982           40.753            -73.981   \n",
       "75%         12.500           -73.968           40.768            -73.965   \n",
       "max        952.000           -70.000           44.732            -70.002   \n",
       "\n",
       "       dropoff_latitude  passenger_count  \n",
       "count       4878498.000      4878498.000  \n",
       "mean             40.751            1.690  \n",
       "std               0.037            1.314  \n",
       "min              37.031            1.000  \n",
       "25%              40.736            1.000  \n",
       "50%              40.754            1.000  \n",
       "75%              40.768            2.000  \n",
       "max              44.728          208.000  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "taxi.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To be quick, let's create a baseline model, without Machine learning, just a simple rate calculation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "train, test = train_test_split(taxi, test_size=0.3, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import shutil\n",
    "\n",
    "def distance_between(lat1, lon1, lat2, lon2):\n",
    "  # Haversine formula to compute distance \n",
    "  dist = np.degrees(np.arccos(np.sin(np.radians(lat1)) * np.sin(np.radians(lat2)) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon2 - lon1)))) * 60 * 1.515 * 1.609344\n",
    "  return dist\n",
    "\n",
    "def estimate_distance(df):\n",
    "  return distance_between(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])\n",
    "\n",
    "def compute_rmse(actual, predicted):\n",
    "  return np.sqrt(np.mean((actual - predicted)**2))\n",
    "\n",
    "def print_rmse(df, rate, name):\n",
    "  print(\"{1} RMSE = {0}\".format(compute_rmse(df['fare_amount'], rate * estimate_distance(df)), name))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\SusanLi\\AppData\\Local\\Continuum\\anaconda3\\lib\\site-packages\\ipykernel_launcher.py:6: RuntimeWarning: invalid value encountered in arccos\n",
      "  \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Rate = $2.565558644556365/km\n",
      "Train RMSE = 9.900882019987426\n",
      "Test RMSE = 9.911110532387688\n"
     ]
    }
   ],
   "source": [
    "rate = train['fare_amount'].mean() / estimate_distance(train).mean()\n",
    "\n",
    "print(\"Rate = ${0}/km\".format(rate))\n",
    "print_rmse(train, rate, 'Train')\n",
    "print_rmse(test, rate, 'Test')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This baseline model gets us RMSE for test set at $9.91. We expect ML achieve better than this. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Feature engineering\n",
    "\n",
    "1). Extract information from datetime (day of week, month, hour, day). Taxi fares change day/night or on weekdays/holidays.\n",
    "\n",
    "2). The distance from pickup to dropoff. The longer the trip, the higher the price.\n",
    "\n",
    "3). Add columns indicating distance from pickup or dropoff coordinates to JFK. Trips from/to JFK have a flat fare at $52.\n",
    "\n",
    "Getting distance between two points based on latitude and longitude using haversine formula. \n",
    "https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas/29546836#29546836"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "taxi['year'] = taxi.pickup_datetime.dt.year\n",
    "taxi['month'] = taxi.pickup_datetime.dt.month\n",
    "taxi['day'] = taxi.pickup_datetime.dt.day\n",
    "taxi['weekday'] = taxi.pickup_datetime.dt.weekday\n",
    "taxi['hour'] = taxi.pickup_datetime.dt.hour"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fare_amount</th>\n",
       "      <th>pickup_datetime</th>\n",
       "      <th>pickup_longitude</th>\n",
       "      <th>pickup_latitude</th>\n",
       "      <th>dropoff_longitude</th>\n",
       "      <th>dropoff_latitude</th>\n",
       "      <th>passenger_count</th>\n",
       "      <th>year</th>\n",
       "      <th>month</th>\n",
       "      <th>day</th>\n",
       "      <th>weekday</th>\n",
       "      <th>hour</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>4.500</td>\n",
       "      <td>2009-06-15 17:26:21</td>\n",
       "      <td>-73.844</td>\n",
       "      <td>40.721</td>\n",
       "      <td>-73.842</td>\n",
       "      <td>40.712</td>\n",
       "      <td>1</td>\n",
       "      <td>2009</td>\n",
       "      <td>6</td>\n",
       "      <td>15</td>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>16.900</td>\n",
       "      <td>2010-01-05 16:52:16</td>\n",
       "      <td>-74.016</td>\n",
       "      <td>40.711</td>\n",
       "      <td>-73.979</td>\n",
       "      <td>40.782</td>\n",
       "      <td>1</td>\n",
       "      <td>2010</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5.700</td>\n",
       "      <td>2011-08-18 00:35:00</td>\n",
       "      <td>-73.983</td>\n",
       "      <td>40.761</td>\n",
       "      <td>-73.991</td>\n",
       "      <td>40.751</td>\n",
       "      <td>2</td>\n",
       "      <td>2011</td>\n",
       "      <td>8</td>\n",
       "      <td>18</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7.700</td>\n",
       "      <td>2012-04-21 04:30:42</td>\n",
       "      <td>-73.987</td>\n",
       "      <td>40.733</td>\n",
       "      <td>-73.992</td>\n",
       "      <td>40.758</td>\n",
       "      <td>1</td>\n",
       "      <td>2012</td>\n",
       "      <td>4</td>\n",
       "      <td>21</td>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.300</td>\n",
       "      <td>2010-03-09 07:51:00</td>\n",
       "      <td>-73.968</td>\n",
       "      <td>40.768</td>\n",
       "      <td>-73.957</td>\n",
       "      <td>40.784</td>\n",
       "      <td>1</td>\n",
       "      <td>2010</td>\n",
       "      <td>3</td>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   fare_amount     pickup_datetime  pickup_longitude  pickup_latitude  \\\n",
       "0        4.500 2009-06-15 17:26:21           -73.844           40.721   \n",
       "1       16.900 2010-01-05 16:52:16           -74.016           40.711   \n",
       "2        5.700 2011-08-18 00:35:00           -73.983           40.761   \n",
       "3        7.700 2012-04-21 04:30:42           -73.987           40.733   \n",
       "4        5.300 2010-03-09 07:51:00           -73.968           40.768   \n",
       "\n",
       "   dropoff_longitude  dropoff_latitude  passenger_count  year  month  day  \\\n",
       "0            -73.842            40.712                1  2009      6   15   \n",
       "1            -73.979            40.782                1  2010      1    5   \n",
       "2            -73.991            40.751                2  2011      8   18   \n",
       "3            -73.992            40.758                1  2012      4   21   \n",
       "4            -73.957            40.784                1  2010      3    9   \n",
       "\n",
       "   weekday  hour  \n",
       "0        0    17  \n",
       "1        1    16  \n",
       "2        3     0  \n",
       "3        5     4  \n",
       "4        1     7  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "taxi.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "from math import radians, cos, sin, asin, sqrt\n",
    "import numpy as np\n",
    "\n",
    "def haversine_np(lon1, lat1, lon2, lat2):\n",
    "    \"\"\"\n",
    "    Calculate the great circle distance between two points\n",
    "    on the earth (specified in decimal degrees)\n",
    "\n",
    "    All args must be of equal length.    \n",
    "\n",
    "    \"\"\"\n",
    "    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])\n",
    "\n",
    "    dlon = lon2 - lon1\n",
    "    dlat = lat2 - lat1\n",
    "\n",
    "    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2\n",
    "\n",
    "    c = 2 * np.arcsin(np.sqrt(a))\n",
    "    km = 6371 * c  # 6371 is Radius of earth in kilometers. Use 3956 for miles\n",
    "    return km\n",
    "\n",
    "taxi['distance'] = haversine_np(taxi['pickup_latitude'], taxi['pickup_longitude'], taxi['dropoff_latitude'] , taxi['dropoff_longitude'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fare_amount</th>\n",
       "      <th>pickup_datetime</th>\n",
       "      <th>pickup_longitude</th>\n",
       "      <th>pickup_latitude</th>\n",
       "      <th>dropoff_longitude</th>\n",
       "      <th>dropoff_latitude</th>\n",
       "      <th>passenger_count</th>\n",
       "      <th>year</th>\n",
       "      <th>month</th>\n",
       "      <th>day</th>\n",
       "      <th>weekday</th>\n",
       "      <th>hour</th>\n",
       "      <th>distance</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>4.500</td>\n",
       "      <td>2009-06-15 17:26:21</td>\n",
       "      <td>-73.844</td>\n",
       "      <td>40.721</td>\n",
       "      <td>-73.842</td>\n",
       "      <td>40.712</td>\n",
       "      <td>1</td>\n",
       "      <td>2009</td>\n",
       "      <td>6</td>\n",
       "      <td>15</td>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "      <td>0.410</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>16.900</td>\n",
       "      <td>2010-01-05 16:52:16</td>\n",
       "      <td>-74.016</td>\n",
       "      <td>40.711</td>\n",
       "      <td>-73.979</td>\n",
       "      <td>40.782</td>\n",
       "      <td>1</td>\n",
       "      <td>2010</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>16</td>\n",
       "      <td>4.629</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5.700</td>\n",
       "      <td>2011-08-18 00:35:00</td>\n",
       "      <td>-73.983</td>\n",
       "      <td>40.761</td>\n",
       "      <td>-73.991</td>\n",
       "      <td>40.751</td>\n",
       "      <td>2</td>\n",
       "      <td>2011</td>\n",
       "      <td>8</td>\n",
       "      <td>18</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>1.001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7.700</td>\n",
       "      <td>2012-04-21 04:30:42</td>\n",
       "      <td>-73.987</td>\n",
       "      <td>40.733</td>\n",
       "      <td>-73.992</td>\n",
       "      <td>40.758</td>\n",
       "      <td>1</td>\n",
       "      <td>2012</td>\n",
       "      <td>4</td>\n",
       "      <td>21</td>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "      <td>0.910</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.300</td>\n",
       "      <td>2010-03-09 07:51:00</td>\n",
       "      <td>-73.968</td>\n",
       "      <td>40.768</td>\n",
       "      <td>-73.957</td>\n",
       "      <td>40.784</td>\n",
       "      <td>1</td>\n",
       "      <td>2010</td>\n",
       "      <td>3</td>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>1.361</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   fare_amount     pickup_datetime  pickup_longitude  pickup_latitude  \\\n",
       "0        4.500 2009-06-15 17:26:21           -73.844           40.721   \n",
       "1       16.900 2010-01-05 16:52:16           -74.016           40.711   \n",
       "2        5.700 2011-08-18 00:35:00           -73.983           40.761   \n",
       "3        7.700 2012-04-21 04:30:42           -73.987           40.733   \n",
       "4        5.300 2010-03-09 07:51:00           -73.968           40.768   \n",
       "\n",
       "   dropoff_longitude  dropoff_latitude  passenger_count  year  month  day  \\\n",
       "0            -73.842            40.712                1  2009      6   15   \n",
       "1            -73.979            40.782                1  2010      1    5   \n",
       "2            -73.991            40.751                2  2011      8   18   \n",
       "3            -73.992            40.758                1  2012      4   21   \n",
       "4            -73.957            40.784                1  2010      3    9   \n",
       "\n",
       "   weekday  hour  distance  \n",
       "0        0    17     0.410  \n",
       "1        1    16     4.629  \n",
       "2        3     0     1.001  \n",
       "3        5     4     0.910  \n",
       "4        1     7     1.361  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "taxi.head()"
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
       "<matplotlib.figure.Figure at 0x17ce29a5b00>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize = (14, 4))\n",
    "n, bins, patches = plt.hist(taxi.distance, 1000, facecolor='blue', alpha=0.75)\n",
    "plt.xlabel('distance')\n",
    "plt.title('Histogram of ride distance')\n",
    "plt.show();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count   4878492.000\n",
       "mean          2.724\n",
       "std           4.275\n",
       "min           0.000\n",
       "25%           0.853\n",
       "50%           1.552\n",
       "75%           2.831\n",
       "max         424.674\n",
       "Name: distance, dtype: float64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "taxi['distance'].describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The minimum distance is 0, we will remove all 0 distance."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "taxi = taxi.loc[taxi['distance'] > 0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Official NYC yellow taxis have a flat rate fee from JFK to Manhattan for $52 (plus tolls and tip), Add columns indicating distance from pickup or dropoff coordinates to JFK."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "JFK_coord = (40.6413, -73.7781)\n",
    "\n",
    "pickup_JFK = haversine_np(taxi['pickup_latitude'], taxi['pickup_longitude'], JFK_coord[0], JFK_coord[1]) \n",
    "dropoff_JFK = haversine_np(JFK_coord[0], JFK_coord[1], taxi['dropoff_latitude'], taxi['dropoff_longitude'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "taxi['JFK_distance'] = pd.concat([pickup_JFK, dropoff_JFK], axis=1).min(axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count   4826440.000\n",
       "mean         21.001\n",
       "std           4.558\n",
       "min           0.021\n",
       "25%          20.274\n",
       "50%          22.062\n",
       "75%          23.150\n",
       "max         455.905\n",
       "Name: JFK_distance, dtype: float64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "taxi['JFK_distance'].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fare_amount</th>\n",
       "      <th>pickup_datetime</th>\n",
       "      <th>pickup_longitude</th>\n",
       "      <th>pickup_latitude</th>\n",
       "      <th>dropoff_longitude</th>\n",
       "      <th>dropoff_latitude</th>\n",
       "      <th>passenger_count</th>\n",
       "      <th>year</th>\n",
       "      <th>month</th>\n",
       "      <th>day</th>\n",
       "      <th>weekday</th>\n",
       "      <th>hour</th>\n",
       "      <th>distance</th>\n",
       "      <th>JFK_distance</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>4.500</td>\n",
       "      <td>2009-06-15 17:26:21</td>\n",
       "      <td>-73.844</td>\n",
       "      <td>40.721</td>\n",
       "      <td>-73.842</td>\n",
       "      <td>40.712</td>\n",
       "      <td>1</td>\n",
       "      <td>2009</td>\n",
       "      <td>6</td>\n",
       "      <td>15</td>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "      <td>0.410</td>\n",
       "      <td>7.397</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>16.900</td>\n",
       "      <td>2010-01-05 16:52:16</td>\n",
       "      <td>-74.016</td>\n",
       "      <td>40.711</td>\n",
       "      <td>-73.979</td>\n",
       "      <td>40.782</td>\n",
       "      <td>1</td>\n",
       "      <td>2010</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>16</td>\n",
       "      <td>4.629</td>\n",
       "      <td>22.787</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5.700</td>\n",
       "      <td>2011-08-18 00:35:00</td>\n",
       "      <td>-73.983</td>\n",
       "      <td>40.761</td>\n",
       "      <td>-73.991</td>\n",
       "      <td>40.751</td>\n",
       "      <td>2</td>\n",
       "      <td>2011</td>\n",
       "      <td>8</td>\n",
       "      <td>18</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>1.001</td>\n",
       "      <td>23.054</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7.700</td>\n",
       "      <td>2012-04-21 04:30:42</td>\n",
       "      <td>-73.987</td>\n",
       "      <td>40.733</td>\n",
       "      <td>-73.992</td>\n",
       "      <td>40.758</td>\n",
       "      <td>1</td>\n",
       "      <td>2012</td>\n",
       "      <td>4</td>\n",
       "      <td>21</td>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "      <td>0.910</td>\n",
       "      <td>23.415</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.300</td>\n",
       "      <td>2010-03-09 07:51:00</td>\n",
       "      <td>-73.968</td>\n",
       "      <td>40.768</td>\n",
       "      <td>-73.957</td>\n",
       "      <td>40.784</td>\n",
       "      <td>1</td>\n",
       "      <td>2010</td>\n",
       "      <td>3</td>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>1.361</td>\n",
       "      <td>20.336</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   fare_amount     pickup_datetime  pickup_longitude  pickup_latitude  \\\n",
       "0        4.500 2009-06-15 17:26:21           -73.844           40.721   \n",
       "1       16.900 2010-01-05 16:52:16           -74.016           40.711   \n",
       "2        5.700 2011-08-18 00:35:00           -73.983           40.761   \n",
       "3        7.700 2012-04-21 04:30:42           -73.987           40.733   \n",
       "4        5.300 2010-03-09 07:51:00           -73.968           40.768   \n",
       "\n",
       "   dropoff_longitude  dropoff_latitude  passenger_count  year  month  day  \\\n",
       "0            -73.842            40.712                1  2009      6   15   \n",
       "1            -73.979            40.782                1  2010      1    5   \n",
       "2            -73.991            40.751                2  2011      8   18   \n",
       "3            -73.992            40.758                1  2012      4   21   \n",
       "4            -73.957            40.784                1  2010      3    9   \n",
       "\n",
       "   weekday  hour  distance  JFK_distance  \n",
       "0        0    17     0.410         7.397  \n",
       "1        1    16     4.629        22.787  \n",
       "2        3     0     1.001        23.054  \n",
       "3        5     4     0.910        23.415  \n",
       "4        1     7     1.361        20.336  "
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "taxi.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "del taxi['pickup_datetime']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "y = taxi['fare_amount']\n",
    "X = taxi.drop(columns=['fare_amount'])\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "lr = LinearRegression()\n",
    "lr.fit(X_train, y_train)\n",
    "y_pred = lr.predict(X_test)"
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
      "Test RMSE: 6.037\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import mean_squared_error\n",
    "\n",
    "print(\"Test RMSE: %.3f\" % mean_squared_error(y_test, y_pred) ** 0.5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Random Forest Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\SusanLi\\AppData\\Local\\Continuum\\anaconda3\\lib\\site-packages\\sklearn\\ensemble\\weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.\n",
      "  from numpy.core.umath_tests import inner1d\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test RMSE: 5.665\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor\n",
    "\n",
    "rf = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)\n",
    "rf.fit(X_train, y_train)\n",
    "y_pred = rf.predict(X_test)\n",
    "print(\"Test RMSE: %.3f\" % mean_squared_error(y_test, y_pred) ** 0.5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### LightGBM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "import lightgbm as lgb\n",
    "\n",
    "params = {\n",
    "        'learning_rate': 0.75,\n",
    "        'application': 'regression',\n",
    "        'max_depth': 3,\n",
    "        'num_leaves': 100,\n",
    "        'verbosity': -1,\n",
    "        'metric': 'RMSE',\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_set = lgb.Dataset(X_train, y_train, silent=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "lb = lgb.train(params, train_set = train_set, num_boost_round=300)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = lb.predict(X_test, num_iteration = lb.best_iteration)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test RMSE: 3.729\n"
     ]
    }
   ],
   "source": [
    "print(\"Test RMSE: %.3f\" % mean_squared_error(y_test, y_pred) ** 0.5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### A Baseline Regression Model with Keras"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import Sequential\n",
    "from keras.layers import Dense\n",
    "from keras.wrappers.scikit_learn import KerasRegressor\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.model_selection import KFold\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.pipeline import Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "def baseline_model():\n",
    "    # create model\n",
    "    model = Sequential()\n",
    "    model.add(Dense(12, input_dim=12, kernel_initializer='normal', activation='relu'))\n",
    "    model.add(Dense(1, kernel_initializer='normal'))\n",
    "    # Compile model\n",
    "    model.compile(loss='mean_squared_error', optimizer='adam')\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "seed = 7\n",
    "np.random.seed(seed)\n",
    "# evaluate model with standardized dataset\n",
    "estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Results: -29.41 (1.49) MSE\n"
     ]
    }
   ],
   "source": [
    "kfold = KFold(n_splits=10, random_state=seed)\n",
    "results = cross_val_score(estimator, X.values, y.values, cv=kfold, n_jobs=1)\n",
    "print(\"Results: %.2f (%.2f) MSE\" % (results.mean(), results.std()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RMSE: 1.2203621092924317\n"
     ]
    }
   ],
   "source": [
    "print(\"RMSE:\", np.sqrt(results.std()))"
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
   "version": "3.6.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

# When Will They Fix It?
## Predicting 311 Response Times in Chicago

[Chicago 311](https://311.chicago.gov/s/?language=en_US) is the city-run phone
and web service where Chicago residents can place non-emergency requests
concerning issues such as broken traffic signals, potholes, and stray animals.
After a Chicago resident submits a 311 service request, the request's status is
visible on an online portal, along with the city's estimate for how long the
request will take to fulfill. However, the city provides little transparency
into how the time-to-complete estimate is generated or how accurate it may be,
beyond a statement that the “estimated completion time may vary depending on
request volume, time of year, and other factors.”

In this project, we have trained models that (1) estimate the time needed to
complete a 311 request, and (2) whether a request will be completed within a
one-week timeframe.

## Structure

__clean_data.py__ filters and processes the [311 Service Requests](https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy) dataset from the city of Chicago.
Note that due to the size of this dataset, it has not been checked in to this repo.

__pipeline.py__ contains helper functions that aid in processing data and
training and evaluating models.

__model_impl.py__ includes NumPy implementations of ridge and principal
components regression models.

__train_models.py__ splits the processed data into train, validation, and test
sets, and runs a grid search over pre-selected models and hyperparameters to
identify the best models for both the regression and classification tasks
(identified in the introduction). The best models, along with their associated
test error, are saved to __best_regr_model.pkl__ and __best_clsf_model.pkl__.

The __pickle_files__ folder includes the processed data that is the input to
__train_models.py__, and the 'best models' that are the output.

__311_request_types.json__ includes a list of all possible 311 request types
(as 311 request type is a key feature in both models).

## Authors
The authors of this repository are Mike Feldman and Lily Grier, both graduate
students at the University of Chicago.

## License
[MIT](https://choosealicense.com/licenses/mit/)

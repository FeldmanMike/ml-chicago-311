'''
Clean data before training models
'''
import pandas as pd


def clean_311_data():
    '''
    Clean raw Chicago 311 data and generate pickle file for model building

    Note: Chicago 311 data file has ~4M rows, so has not been pushed
    to this repo
    '''
    print('Reading in CSV...')
    chi_311 = pd.read_csv('./raw_data/chicago_311_requests.csv')

    # Filter dataframe to exclude info-only calls
    print('Filtering out info-only calls dataframe...')
    chi_311_filtered = chi_311[chi_311['SR_TYPE'] !=
                                '311 INFORMATION ONLY CALL']

    chi_311_filtered = chi_311_filtered[['SR_NUMBER', 'SR_TYPE', 'STATUS',
                                         'CREATED_DATE', 'CLOSED_DATE',
                                         'DUPLICATE', 'LEGACY_RECORD',
                                         'LEGACY_SR_NUMBER',
                                         'PARENT_SR_NUMBER', 'WARD',
                                         'CREATED_HOUR', 'CREATED_DAY_OF_WEEK',
                                         'CREATED_MONTH']]

    # Filter out legacy records
    print('Filter out legacy records...')
    chi_311_filtered = chi_311_filtered[chi_311_filtered['LEGACY_RECORD']
                                        == False]
    chi_311_filtered = chi_311_filtered.drop(columns=['LEGACY_RECORD',
                                                      'LEGACY_SR_NUMBER'])

    # Filter out rows with no ward
    print('Filter out rows with no ward...')
    chi_311_filtered = chi_311_filtered[chi_311_filtered['WARD'].notna()]

    # Add columns with number of 'children' a given request has
    print('Add columns with number of children per request...')
    parent_groups = pd.DataFrame(chi_311_filtered['PARENT_SR_NUMBER'].
                                    value_counts())

    parent_groups = parent_groups.rename(columns={'PARENT_SR_NUMBER':
                                                  'NUM_CHILDREN'})

    chi_311_filtered = chi_311_filtered.merge(parent_groups, how='left',
                                              left_on='SR_NUMBER',
                                              right_index=True)

    chi_311_filtered = chi_311_filtered[chi_311_filtered['DUPLICATE'] == False]
    chi_311_filtered = chi_311_filtered.drop(columns=['DUPLICATE',
                                                      'PARENT_SR_NUMBER'])

    print('Make dummy columns...')
    chi_311_filtered = pd.get_dummies(chi_311_filtered,
                                      columns=['WARD', 'CREATED_HOUR', 
                                               'CREATED_DAY_OF_WEEK',
                                               'CREATED_MONTH', 'SR_TYPE'])

    # Filter out requests resolved in less than 10 minutes (i.e., requests that
    # don't require someone dispatched to solve).

    # Note this was a somewhat arbitrary choice made. A large number of
    # requests were resolved in 0 seconds (instantaneously) so we had to
    # filter those out. Many were resolved in less than 1 minute as well. The
    # number resolved immediately seemed to level off at the 10-minute mark.
    # Given we wanted to filter out 'info-only' calls and wanted to measure
    # response time for issues that required active assistance, this was
    # sufficient information for this filter decision
    print('Filter out requests resolved in less than 10 minutes...')
    chi_311_filtered['CREATED_DATE'] = pd.to_datetime(
                                        chi_311_filtered['CREATED_DATE'],
                                        format='%m/%d/%Y %I:%M:%S %p')

    chi_311_filtered['CLOSED_DATE'] = pd.to_datetime(
                                        chi_311_filtered['CLOSED_DATE'],
                                        format='%m/%d/%Y %I:%M:%S %p')

    chi_311_filtered['time_to_close'] = chi_311_filtered['CLOSED_DATE'] - \
                                            chi_311_filtered['CREATED_DATE']

    chi_311_filtered['time_to_close_sec'] = chi_311_filtered[
                                            'time_to_close'].dt.total_seconds()

    chi_311_filtered = chi_311_filtered.drop(columns=['time_to_close'])
    chi_311_filtered = chi_311_filtered[chi_311_filtered['time_to_close_sec']
                                        >= 600]

    # Created time to close hour column
    print('Create time to close hour column...')
    chi_311_filtered['time_to_close_hr'] = chi_311_filtered[
                                            'time_to_close_sec'] / 3600

    # Set new index
    print('Set new index...')
    chi_311_filtered = chi_311_filtered.set_index('SR_NUMBER')

    # Filtered out uncompleted (note this will bias predicted response times
    # downward - would need another model to predict probability request
    # would be completed)
    print('Filter out uncompleted...')
    chi_311_filtered = chi_311_filtered[chi_311_filtered['STATUS'] ==
                                        'Completed']
    
    # Drop other unneeded columns
    print('Drop unneeded columns...')
    chi_311_filtered = chi_311_filtered.drop(columns=['STATUS', 'CREATED_DATE',
                                                      'CLOSED_DATE',
                                                      'time_to_close_sec'])

    print('Columns are now...', chi_311_filtered.columns)

    # Fill NAs as needed
    print('Fill NAs as needed...')
    chi_311_filtered['NUM_CHILDREN'] = chi_311_filtered[
                                        'NUM_CHILDREN'].fillna(0)

    # Send cleaned file to pickle
    print('Send cleaned file to pickle...')
    chi_311_filtered.to_pickle("./pickle_files/chi_311.pkl")


if __name__ == '__main__':
    clean_311_data()

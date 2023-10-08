warning off
data=readtable('/Users/saiharshinigarapati/Desktop/AirQuality.csv')
n=5;
disp(data(1:n,:));

data.CO_GT_=[];
disp(data(1:n,:));
%convert string to float and remove ' to .
% Specify the columns to be processed
columns_to_convert = {'C6H6_GT_', 'T', 'RH', 'AH'}; % Replace with your column names

% Loop through the columns and apply the conversion
for i = 1:length(columns_to_convert)
    column_name = columns_to_convert{i};
    
    % Replace ',' with '.' in the specified column
    data.(column_name) = strrep(data.(column_name), ',', '.');
    
    % Convert the column to a numeric array
    numeric_array = str2double(data.(column_name));
    
    % Check if the numeric array has the same number of rows as the table
    if numel(numeric_array) == height(data)
        % Convert the numeric array to a cell array of strings
        cell_array_of_strings = cellstr(num2str(numeric_array));
        
        % Assign the cell array of strings back to the table
        data.(column_name) = cell_array_of_strings;
    else
        fprintf('Error: Numeric array dimensions do not match the table height for column %s\n', column_name);
    end
end

% Display the modified table
disp(data);
% Assuming you have a table named 'data' with your data

% Specify the null data value (-200 in this case)
null_value = -200;

% Loop through the columns and replace null values with NaN
for i = 1:size(data, 2) % Loop through all columns
    column_data = data.(i); % Get the data from the current column
    
    if isnumeric(column_data)
        % For numeric columns, replace null values with NaN
        column_data(column_data == null_value) = NaN;
    elseif isa(column_data, 'datetime')
        % For datetime columns, replace null values with NaT
        column_data(isnat(column_data)) = NaT;
    end
    
    data.(i) = column_data; % Update the column in the table
end

% Display the modified table
disp(data);


% Assuming you have a table named 'data' with 'Date' and 'Time' columns

% Convert the 'Date' column to datetime (assuming 'dd.mm.yyyy' format)
data.Date = datetime(data.Date, 'InputFormat', 'dd.MM.yyyy', 'Format', 'dd.MM.yyyy');

% Replace periods ('.') with colons (':') in the 'Time' column
data.Time = strrep(data.Time, '.', ':');

% Convert the 'Time' column to datetime with 'HH:mm:ss' format
data.Time = datetime(data.Time, 'InputFormat', 'HH:mm:ss', 'Format', 'HH:mm:ss');

% Display the modified table
disp(data);




data.NMHC_GT_=[]


% Define the names of columns to convert to numeric
columns_to_convert = {'C6H6_GT_', 'T', 'RH', 'AH'};

for i = 1:length(columns_to_convert)
    column_name = columns_to_convert{i};
    
    % Get the cell column data by column name
    cell_column_data = data.(column_name);
    
    % Remove non-numeric characters (assuming numeric values are stored as strings)
    cleaned_data = regexprep(cell_column_data, '[^0-9.]', '');
    
    % Convert the cleaned data to numeric values using str2double
    numeric_values = str2double(cleaned_data);
    
    % Replace the original cell column with the numeric values
    data.(column_name) = numeric_values;
end

% Display the modified table with numeric values
disp(data);




% Define the range of columns to be processed (columns 2 to 13)
columns_to_process = 3:13;

% Specify the scale for IQR (e.g., 2 for Normal Distributions)
scale = 2;

% Calculate quartiles and IQR for each numeric column
Q1 = quantile(data{:, columns_to_process}, 0.25);
Q3 = quantile(data{:, columns_to_process}, 0.75);
IQR = Q3 - Q1;

% Calculate lower and upper limits for outliers
lower_lim = Q1 - scale * IQR;
upper_lim = Q3 + scale * IQR;

% Detect outliers for each numeric column
lower_outliers = data{:, columns_to_process} < lower_lim;
upper_outliers = data{:, columns_to_process} > upper_lim;

% Check the resulting outliers (represented below as non-null values)
outliers = data{:, columns_to_process}(lower_outliers | upper_outliers);

% Display the rows and columns with outliers
disp("Rows and Columns with Outliers:");
disp(outliers);

% Define the range of columns to be processed (columns 2 to 13)
columns_to_process = 3:13;

% Specify the scale for IQR (e.g., 2 for Normal Distributions)
scale = 2;

% Calculate quartiles and IQR for each numeric column
Q1 = quantile(data{:, columns_to_process}, 0.25);
Q3 = quantile(data{:, columns_to_process}, 0.75);
IQR = Q3 - Q1;

% Calculate lower and upper limits for outliers
lower_lim = Q1 - scale * IQR;
upper_lim = Q3 + scale * IQR;

% Detect outliers for each numeric column
lower_outliers = data{:, columns_to_process} < lower_lim;
upper_outliers = data{:, columns_to_process} > upper_lim;

% Determine rows with outliers
rows_with_outliers = any(lower_outliers | upper_outliers, 2);

% Create a new table without the outliers
data_without_outliers = data(~rows_with_outliers, :);

% Display information about the new table
disp("Information about the New Table without Outliers:");
disp(data_without_outliers.Properties.Description);  % Display table description
disp(data_without_outliers.Properties.VariableNames); % Display variable names
disp(size(data_without_outliers));                   % Display table size

data_without_outliers.NOx_GT_=[]
data_without_outliers.NO2_GT_=[]



% Remove rows with NaN values from all columns
data_filt = rmmissing(data_without_outliers);

% Display information about the filtered table
disp("Information about the Filtered Table:");
disp(data_filt.Properties.Description);  % Display table description
disp(data_filt.Properties.VariableNames); % Display variable names
disp(size(data_filt));                   % Display table size
           % Display table size

% Add a column with week days
dayNames = {'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'};
dayValues = weekday(data_filt.Date);  % Get numeric day values (1-7)

% Check if the number of rows matches
if numel(dayValues) == height(data_filt)
    % Convert numeric day values to cell array of day names
    dayNamesCell = dayNames(dayValues)';
    data_filt.WeekDay = categorical(dayNamesCell, dayNames);
    
    % Rearrange columns
    newOrder = [1, size(data_filt, 2), 2:11];  % Define the new column order
    data_filt = data_filt(:, newOrder);  % Rearrange the columns
    
    % Display the first 10 rows of the modified table
    disp(head(data_filt, 10));
else
    disp("Error: Number of rows in dayValues does not match the table.");
end

data_filt.C6H6_GT_=[]


% Assuming data_filt is your table

% Define the training and testing proportions
trainProportion = 0.8;  % 80% for training
testProportion = 1 - trainProportion;  % 20% for testing

% Create a random partition
rng(42);  % Set the random seed for reproducibility
cvp = cvpartition(height(data_filt), 'HoldOut', testProportion);

% Split the data into training and testing sets
X_train = data_filt(cvp.training, :);  % 80% for training
X_test = data_filt(cvp.test, :);  % 20% for testing

% Display the sizes of the training and testing sets
disp(['Training Set Size: ', num2str(height(X_train))]);
disp(['Testing Set Size: ', num2str(height(X_test))]);

% Remove the 'Date' and 'Time' variables from the predictor matrix
X_train = X_train(:, ~ismember(X_train.Properties.VariableNames, {'Date', 'Time'}));

% Define the target variable (PT08_S1_CO_)
Y_train = X_train.PT08_S1_CO_;
Y_test=X_test.PT08_S1_CO_;
X_train = X_train(:, ~ismember(X_train.Properties.VariableNames, 'PT08_S1_CO_'));

% Create a Random Forest regression model with 100 trees
numTrees = 100;
model = TreeBagger(numTrees, X_train, Y_train, 'Method', 'regression');

% Now you have a trained Random Forest regression model


% Assuming X_test contains your test data
X_test = X_test(:, ~ismember(X_test.Properties.VariableNames, {'Date', 'Time'}));
Y_pred = predict(model, X_test);
R2_score_without_seasons = 1 - sum((Y_test - Y_pred).^2) / sum((Y_test - mean(Y_test)).^2);

data_filt.Season = arrayfun(@extractSeason, data_filt.Date, 'UniformOutput', false);



% Extract the target variable (PT08_S1_CO_)
Y_2 = data_filt.PT08_S1_CO_;

% Check if 'Season' is a variable in your table
if ismember('Season', data_filt.Properties.VariableNames)
    % Extract the 'Season' column
    SeasonColumn = data_filt.Season;

    % Convert the 'Season' column to a categorical variable
    SeasonCategorical = categorical(SeasonColumn);

    % Create dummy variables for the 'Season' column
    SeasonDummies = dummyvar(SeasonCategorical);

    % Create a table from the dummy variables
    SeasonDummiesTable = array2table(SeasonDummies);

    % Rename the dummy variable columns
    dummyVarNames = strcat('Season_', cellstr(categories(SeasonCategorical)))';
    SeasonDummiesTable.Properties.VariableNames = dummyVarNames;

    % Concatenate the dummy variables to the original table
    data_encoded = [data_filt SeasonDummiesTable];

    % Remove the original 'Season' column
    data_encoded.Season = [];

    % Define the random seed for reproducibility
    rng(42);

    % Define the training and testing proportions
    trainProportion = 0.8;  % 80% for training
    testProportion = 1 - trainProportion;  % 20% for testing

    % Create a random partition
    cvp = cvpartition(height(data_encoded), 'HoldOut', testProportion);

    % Split the data into training and testing sets
    data_encoded_train = data_encoded(cvp.training, :);  % 80% for training
    data_encoded_test = data_encoded(cvp.test, :);  % 20% for testing
    Y_2_train = data_encoded_train.PT08_S1_CO_;  % Target variable for training
    Y_2_test = data_encoded_test.PT08_S1_CO_;  % Target variable for testing

    % Display the sizes of the training and testing sets
    disp(['Training Set Size: ', num2str(height(data_encoded_train))]);
    disp(['Testing Set Size: ', num2str(height(data_encoded_test))]);
else
    disp('The ''Season'' variable does not exist in the table.');
end


% Assuming you have X_2_train and Y_2_train from your previous code

% Remove the 'Date' and 'Time' columns from the predictor matrix
X_2_train = data_encoded_train(:, ~ismember(data_encoded_train.Properties.VariableNames, {'Date','Time'}));

% Create a Random Forest regression model with 100 trees
modelo_randomforest_2 = TreeBagger(100, X_2_train, Y_2_train, 'Method', 'regression');

X_2_test = data_encoded_test(:, ~ismember(data_encoded_test.Properties.VariableNames, {'Date','Time'}));

% Perform predictions with the trained Random Forest model with Seasons
pred_randomforest_2 = modelo_randomforest_2.predict(X_2_test);

% Calculate the R² score for the model with Seasons
R2_score_with_seasons = 1 - sum((Y_2_test - pred_randomforest_2).^2) / sum((Y_2_test - mean(Y_2_test)).^2);


modelFilePath = '/Users/saiharshinigarapati/Desktop/model.mat';

% Save the Random Forest model to a .mat file
save(modelFilePath, 'model','-mat');



fprintf('Regression Model without Seasons: R²=%.2f\n', R2_score_without_seasons);
fprintf('Regression Model with Seasons: R²=%.2f\n', R2_score_with_seasons);

% Define a function to extract season
function season = extractSeason(date)
    year = date.Year;
    springStart = datetime(year, 3, 21);
    summerStart = datetime(year, 6, 21);
    autumnStart = datetime(year, 9, 23);
    winterStart = datetime(year, 12, 21);
    
    if date >= springStart && date < summerStart
        season = 'spring';
    elseif date >= summerStart && date < autumnStart
        season = 'summer';
    elseif date >= autumnStart && date < winterStart
        season = 'autumn';
    else
        season = 'winter';
    end
end




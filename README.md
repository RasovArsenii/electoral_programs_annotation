This component provides a text analysis method that uses a Tf-Idf matrix and SVM to identify the political direction of each particular sentence.
By default, the component contains a model trained on Manifesto data, which provides accuracy of up to 63% on the same as Manifesto data. 

Also you can use our visualization fucntion that show how accurate computer labels data, according to human labeling.
That fuctions based on Pearson correlation coefficient.

Our component can work with Manifesto database directly, using their api.
You need to get api key on their website https://manifesto-project.wzb.eu/ and insert it in ManifestoAPI.py in component folder to use that feature.

Working with our component may take a lot of time(actually it depends on your computing power and amount of data).
For example on Intel Hexacore 4.3 GHz and 117 000 sentences fit and test takes around 45 minutes. 

###########################################################################################################################################################################

										Examples

###########################################################################################################################################################################
    # Import some libs.
>>> import electoral_programs_annotation
>>> import pickle
>>> import pandas as pd

    # Create exemplar of class.
>>> test = electoral_programs_annotation.VSVM()

    # Download data (check params and suit it for your data).											 
>>> data = pd.read_csv('labeled_data.csv', sep=' , ', header=['document_name','data','label'], encoding = 'utf-8 sig', engine='python')  

    # Call method that creates Tf-Idf matrix and fits sci-kit SVM model.
    # It takes one list with sentences and another with labels. Also you need to find some list with stop-words for your language.
    # You can use nltk library that provides it for a lot of languages.
>>> test.fit_transform(data['data'], data['label'], stopwords.words('english'))

    # Test your new model
    # Our component splits your data into test and train parts by itself so You can use test.x_test and test.y_test to reach test data   
    # If the model is not found, use "test.upload_model()"
>>> tmodel = test.model()
>>> tmodel.score(test.x_test, test.y_test)

    # Get a model prediction:
    #
    # This (list of predictions)
>>> prediction = tmodel.predict(test.x_test)
>>> pred_data = DataFrame(data['data'], columns=["sentence"])
>>> data.join(DataFrame(prediction, columns=["prediction"]))
    #
    # Or this (vectorization and prediction from scratch - you can use this on new texts without fitting a model)
>>> another_pred_data = test.predict(data['data'])
    #
    # Or else you can get a more advanced prediction:
    # (the component also contains the "predict_proba" method, which is used in the same way as the "predict" method)
>>> probability = tmodel.predict_proba(test.x_test)
>>> best_pred_data = test.interpretate_proba(test.x_test_text, probability)

    # Visualize results
    # For normal operation, the method must get 'doc_name' and 'code' at the input
>>> data.columns = ['doc_name','data','code']
>>> test.visualize_pearson(data)

    # Save your model for future experiments
    # You can use pickle as very easy library for that
>>> with open('model.pkl', 'wb') as output_file:
>>> 	pickle.dump(test, output_file, protocol=pickle.HIGHEST_PROTOCOL)

    #To load your model use that
>>> with open('model.pkl', 'rb') as input_file:
>>>    	test = pickle.load(input_file)

import os
import pandas as pd
import openai
from openai import OpenAI
import json
import logging


class GPTmethods:
    def __init__(self, model_id='gpt-3.5-turbo'):
        openai.api_key = os.environ.get("OPENAI_API_KEY")  # Access environment variable
        self.model_id = model_id
        self.pre_path = 'Datasets/'

    """
    Create a training and validation JSONL file for GPT fine-tuning
    """

    def create_jsonl(self, data_type, data_set):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(self.pre_path + data_set)
        data = []  # Define a list to store the dictionaries

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            data.append(
                {
                    "messages": [
                        {"role": "system",
                         "content": "You are a crypto expert."},
                        {"role": "user",
                         "content": 'Evaluate the sentiment of the news article. Return your response in JSON format {"sentiment": "negative"} or {"sentiment": "neutral"} or {"sentiment": "positive"}. Article:\n' +
                                    row['text'] + ''},
                        {"role": "assistant", "content": '{"sentiment":"' + str(row['sentiment']) + '"}'}
                    ]
                }  # TODO! Change it!
            )

        output_file_path = self.pre_path + "ft_dataset_gpt_" + data_type + ".jsonl"  # Define the path
        # Write data to the JSONL file
        with open(output_file_path, 'w') as output_file:
            for record in data:
                # Convert the dictionary to a JSON string and write it to the file
                json_record = json.dumps(record)
                output_file.write(json_record + '\n')

        return {"status": True, "data": f"JSONL file '{output_file_path}' has been created."}

    """
    Create a conversation with GPT model
    """

    def gpt_conversation(self, conversation):
        client = OpenAI()
        # response = openai.ChatCompletion.create(
        completion = client.chat.completions.create(
            model=self.model_id,
            messages=conversation
        )
        return completion.choices[0].message

    """
    Clean the response
    """

    # def clean_response(self, response, a_field):
    #     # Search for JSON in the response
    #     start_index = response.find('{')
    #     end_index = response.rfind('}')
    #
    #     if start_index != -1 and end_index != -1:
    #         json_str = response[start_index:end_index + 1]
    #         try:
    #             # Attempt to load the extracted JSON string
    #             json_data = json.loads(json_str)
    #             return {"status": True, "data": json_data}
    #         except json.JSONDecodeError as e:
    #             # If an error occurs during JSON parsing, handle it
    #             logging.error(f"An error occurred while decoding JSON: '{str(e)}'. The input '{a_field}', "
    #                           f"resulted in the following response: {response}")
    #             return {"status": False,
    #                     "data": f"An error occurred while decoding JSON: '{str(e)}'. The input '{a_field}', "
    #                             f"resulted in the following response: {response}"}
    #     else:
    #         logging.error(f"No JSON found in the response. The input '{a_field}', resulted in the "
    #                       f"following response: {response}")
    #         return {"status": False, "data": f"No JSON found in the response. The input '{a_field}', "
    #                                          f"resulted in the following response: {response}"}

    def clean_response(self, response, a_field):
        try:
            # Attempt to parse the JSON string
            response_dict = json.loads(response)
            return {"status": True, "data": response_dict}
        except json.JSONDecodeError:
            # If JSON decoding fails, attempt to correct the format
            try:
                # Correcting the format by adding double quotes around the value
                corrected_response_string = response.replace(':', ':"').replace('}', '"}')
                response_dict = json.loads(corrected_response_string)
                return {"status": True, "data": response_dict}
            except json.JSONDecodeError as e:
                logging.error(f"An error occurred while decoding JSON: '{str(e)}'. The input '{a_field}', "
                              f"resulted in the following response: {response}")
                return {"status": False,
                        "data": f"An error occurred while decoding JSON: '{str(e)}'. The input '{a_field}', "
                                f"resulted in the following response: {response}"}

    """
    Prompt the GPT model to make a prediction
    """

    def gpt_prediction(self, input):
        conversation = []
        conversation.append({'role': 'system',
                             'content': "You are a crypto expert."})  # TODO! Change it!
        conversation.append({'role': 'user',
                             'content': 'Evaluate the sentiment of the news article. Return your response in JSON format {"sentiment": "negative"} or {"sentiment": "neutral"} or {"sentiment": "positive"}. Article:\n' +
                                        input['text'] + ''})  # TODO! Change it!
        conversation = self.gpt_conversation(conversation)  # Get the response from GPT model
        content = conversation.content

        # Clean the response and return
        return self.clean_response(response=content, a_field=input['text'])  # TODO! Change it!

    """
    Make predictions for a specific data_set appending a new prediction_column
    """

    def predictions(self, data_set, prediction_column):

        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.pre_path + data_set)

        # make a copy to _original1
        # Remove the .csv extension from the input file name
        file_name_without_extension = os.path.splitext(os.path.basename(data_set))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original.csv'
        if not os.path.exists(original_file_path):
            os.rename(self.pre_path + data_set, original_file_path)

        # Check if the prediction_column is already present in the header
        if prediction_column not in df.columns:
            # If not, add the column to the DataFrame with pd.NA as the initial value
            df[prediction_column] = pd.NA

            # # Explicitly set the column type to a nullable integer   # TODO! For non-int values omit this if
            # df = df.astype({prediction_column: 'Int64'})

        # Update the CSV file with the new header (if columns were added)
        if prediction_column not in df.columns:
            df.to_csv(self.pre_path + data_set, index=False)

        # Set the dtype of the reason column to object
        # df = df.astype({reason_column: 'object'})

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            # If the prediction column is NaN then proceed to predictions
            if pd.isnull(row[prediction_column]):
                prediction = self.gpt_prediction(input=row)
                if not prediction['status']:
                    print(prediction)
                    break
                else:
                    print(prediction)

                    if prediction['data']['sentiment'] != '':  # TODO! Change it!
                        # Update the DataFrame with the evaluation result
                        df.at[index, prediction_column] = prediction['data']['sentiment']  # TODO! Change it!
                        # for integers only
                        # df.at[index, prediction_column] = int(prediction['data']['sentiment'])  # TODO! Change it!

                        # Update the CSV file with the new evaluation values
                        df.to_csv(self.pre_path + data_set, index=False)
                    else:
                        logging.error(
                            f"No rating instance was found within the data for '{row['text']}', and the "
                            f"corresponding prediction response was: {prediction}.")  # TODO! Change it!
                        return {"status": False,
                                "data": f"No rating instance was found within the data for '{row['text']}', "
                                        f"and the corresponding prediction response was: {prediction}."}  # TODO! Change it!

                # break
            # Add a delay of 5 seconds (reduced for testing)

        # Change the column datatype after processing all predictions to handle 2.0 ratings
        # df[prediction_column] = df[prediction_column].astype('Int64')    # TODO! For non-int values omit this if

        return {"status": True, "data": 'Prediction have successfully been'}

    """
    Upload Dataset for GPT Fine-tuning
    """

    def upload_file(self, dataset):
        upload_file = openai.File.create(
            file=open(dataset, "rb"),
            purpose='fine-tune'
        )
        return upload_file

    """
    Train GPT model
    """

    def train_gpt(self, file_id):
        # https://www.mlq.ai/gpt-3-5-turbo-fine-tuning/
        # https://platform.openai.com/docs/guides/fine-tuning/create-a-fine-tuned-model?ref=mlq.ai
        return openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo")
        # check training status (optional)
        # openai.FineTuningJob.retrieve(file_id)

    """
    Delete Fine-Tuned GPT model
    """

    def delete_finetuned_model(self, model):  # ex. model = ft:gpt-3.5-turbo-0613:personal::84kpHoCN
        return openai.Model.delete(model)

    """
    Cancel Fine-Tuning
    """

    def cancel_gpt_finetuning(self, train_id):  # ex. id = ftjob-3C5lZD1ly5OHHAleLwAqT7Qt
        return openai.FineTuningJob.cancel(train_id)

    """
    Get all Fine-Tuned models and their status
    """

    def get_all_finetuned_models(self):
        return openai.FineTuningJob.list(limit=10)


# Example Usage
# Configure logging to write to a file
logging.basicConfig(filename='error_log.txt', level=logging.ERROR)

# # Instantiate the GPTmethods class
GPT = GPTmethods()
#
# # Create the json file for training
# GPT.create_jsonl(data_type='train', data_set='train_set.csv')  # You have to change the prompt text on each project
#
# # Create the json file for validation
# GPT.create_jsonl(data_type='validation', data_set='validation_set.csv')

# Make predictions before Fine-tuning using the Base Model
# GPT_BM = GPTmethods(model_id='gpt-4-0125-preview')
# GPT_BM.predictions(data_set='test_set.csv', prediction_column='gpt_bm_prediction')

# Make predictions after Fine-tuning using the Fine-tuned (model_id)
# GPT_FT = GPTmethods(model_id='ft:gpt-4-0125-preview:personal::9Aa7kYOh')
# GPT_FT.predictions(data_set='test_set.csv', prediction_column='gpt_ft_prediction')

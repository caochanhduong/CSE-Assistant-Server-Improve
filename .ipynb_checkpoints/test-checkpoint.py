from temp_agent_action_gen import *
from message_handler import *
from agen_response_gen import *



CONSTANT_FILE_PATH = 'constants.json'
with open(CONSTANT_FILE_PATH) as f:
    constants = json.load(f)

file_path_dict = constants['db_file_paths']
DATABASE_FILE_PATH = file_path_dict['database']

database= json.load(open(DATABASE_FILE_PATH,encoding='utf-8'))
state_tracker = StateTracker(database, constants)
dqn_agent = DQNAgent(state_tracker.get_state_size(), constants)
#TEST
if __name__ == '__main__':
    testcase_file= open("logs/testcase_user_action.txt","r",encoding='utf-8')
    output_test_file=open("logs/output_testcase_user_action.txt","w+",encoding='utf-8')
    testcases=testcase_file.readlines()
    num_success_testcases=0
    num_testcases=len(testcases)
    i=1
    output_test_file.write("No\tMessage\tUser Action\n")
    for testcase in testcases:
        # #print(testcase)
        state_tracker = StateTracker(database, constants)
        testcase_without_enter=testcase.replace('\n', '')
        user_action = process_message_to_user_request(testcase_without_enter,state_tracker)
        print(user_action)
        output_test_file.write("{0}\t{1}\t{2}\n".format(i,testcase_without_enter,user_action))
        i+=1
        print(i)
    
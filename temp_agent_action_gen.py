from dqn_agent import DQNAgent
from agent_utils.state_tracker import StateTracker
from agen_response_gen import *
import json
from agent_utils.utils import remove_empty_slots
import sys, os

# CONSTANT_FILE_PATH = 'constants.json'
# with open(CONSTANT_FILE_PATH) as f:
#     constants = json.load(f)

# file_path_dict = constants['db_file_paths']
# DATABASE_FILE_PATH = file_path_dict['database']

# database= json.load(open(DATABASE_FILE_PATH,encoding='utf-8'))
# state_tracker = StateTracker(database, constants)
# dqn_agent = DQNAgent(state_tracker.get_state_size(), constants)

def get_agent_response(state_tracker, dqn_agent, user_action, done=False):
    state_tracker.update_state_user(user_action)
    current_state = state_tracker.get_state(done,user_action)

    _, agent_action = dqn_agent.get_action(current_state)
    state_tracker.update_state_agent(agent_action,user_action)
    assert len(state_tracker.current_request_slots) > 0
    # if agent_action['intent'] == 'match_found':
    #     # print("inform slot match found: {}".format(agent_action['inform_slots']))
    #     agent_action['intent'] = 'inform'
    #     user_request_slot = state_tracker.current_request_slots[0]
    #     # print("user request slot: {}".format(user_request_slot))

    #     inform_value = agent_action['inform_slots'][user_request_slot]
    #     agent_action['inform_slots'].clear()
    #     agent_action['inform_slots'][user_request_slot] = inform_value
        # print("inform slot converted: {}".format(agent_action['inform_slots']))
    return agent_action
# user_act = {'intent': 'inform', 'request_slots': {'name_activity': 'UNK'}, 'inform_slots': {'type_activity': ['chương trình']}}
# agent_act = get_agent_response(state_tracker, dqn_agent, user_act)
# print(response_craft(agent_act, state_tracker))
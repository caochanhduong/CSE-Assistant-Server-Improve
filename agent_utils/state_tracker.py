from .db_query import DBQuery
import numpy as np
from .utils import convert_list_to_dict
from .dialogue_config import all_intents, all_slots, usersim_default_key,agent_inform_slots,agent_request_slots
import copy
import time
from constants import *
from message_handler import check_match_obj

class StateTracker:
    """Tracks the state of the episode/conversation and prepares the state representation for the agent."""

    def __init__(self, database, constants):
        """
        The constructor of StateTracker.
        The constructor of StateTracker which creates a DB query object, creates necessary state rep. dicts, etc. and
        calls reset.
        Parameters:
            database (dict): The database with format dict(long: dict)
            constants (dict): Loaded constants in dict
        """

        self.db_helper = DBQuery(database)
        self.match_key = usersim_default_key
        self.intents_dict = convert_list_to_dict(all_intents)
        self.num_intents = len(all_intents)
        self.slots_dict = convert_list_to_dict(all_slots)
        self.num_slots = len(all_slots)
        self.max_round_num = constants['run']['max_round_num']
        self.none_state = np.zeros(self.get_state_size())
        self.reset()
        self.current_request_slots = []

    def get_state_size(self):
        """Returns the state size of the state representation used by the agent."""

        return 2 * self.num_intents + 7 * self.num_slots + 3 + 13 + self.max_round_num

    def reset(self):
        """Resets current_informs, history and round_num."""

        self.current_informs = {}
        # A list of the dialogues (dicts) by the agent and user so far in the conversation
        self.history = []
        self.round_num = 0
        self.current_request_slots = []

    def print_history(self):
        """Helper function if you want to see the current history action by action."""

        for action in self.history:
            print(action)

    def get_state(self, done=False, user_action):
        """
        Returns the state representation as a numpy array which is fed into the agent's neural network.
        The state representation contains useful information for the agent about the current state of the conversation.
        Processes by the agent to be fed into the neural network. Ripe for experimentation and optimization.
        Parameters:
            done (bool): Indicates whether this is the last dialogue in the episode/conversation. Default: False
        Returns:
            numpy.array: A numpy array of shape (state size,)
        """

        # If done then fill state with zeros
        if done:
            return self.none_state

        user_action = self.history[-1]
        db_results_dict = self.db_helper.get_db_results_for_slots(self.current_informs,user_action)
        last_agent_action = self.history[-2] if len(self.history) > 1 else None
        # print("--------------------------user action")
        # print(user_action)       
        # print("--------------------------last agent action")
        # print(last_agent_action)        

        # Create one-hot of intents to represent the current user action
        user_act_rep = np.zeros((self.num_intents,))
        user_act_rep[self.intents_dict[user_action['intent']]] = 1.0

        # Create bag of inform slots representation to represent the current user action
        user_inform_slots_rep = np.zeros((self.num_slots,))
        for key in user_action['inform_slots'].keys():
            user_inform_slots_rep[self.slots_dict[key]] = 1.0

        # Create bag of request slots representation to represent the current user action
        # EDIT: user request slots should maintain through out the episode
        user_request_slots_rep = np.zeros((self.num_slots,))
        # for key in user_action['request_slots'].keys():
        #     user_request_slots_rep[self.slots_dict[key]] = 1.0
        for key in self.current_request_slots:
            user_request_slots_rep[self.slots_dict[key]] = 1.0

        # Create bag of filled_in slots based on the current_slots
        current_slots_rep = np.zeros((self.num_slots,))
        # print("current inform: {}".format(self.current_informs))
        for key in self.current_informs:
            current_slots_rep[self.slots_dict[key]] = 1.0

        # Encode last agent intent
        agent_act_rep = np.zeros((self.num_intents,))
        if last_agent_action:
            agent_act_rep[self.intents_dict[last_agent_action['intent']]] = 1.0

        # Encode last agent inform slots
        agent_inform_slots_rep = np.zeros((self.num_slots,))
        # print(last_agent_action)
        if last_agent_action:
            for key in last_agent_action['inform_slots'].keys():
                if key in agent_inform_slots:
                    agent_inform_slots_rep[self.slots_dict[key]] = 1.0

        # Encode last agent request slots
        agent_request_slots_rep = np.zeros((self.num_slots,))
        if last_agent_action:
            for key in last_agent_action['request_slots'].keys():
                if key in agent_request_slots:
                    agent_request_slots_rep[self.slots_dict[key]] = 1.0
        # print("------------------------agent_request_slots_rep")
        # print(agent_request_slots_rep)
        # Value representation of the round num
        turn_rep = np.zeros((1,)) + self.round_num / 5.

        # One-hot representation of the round num
        turn_onehot_rep = np.zeros((self.max_round_num,))
        turn_onehot_rep[self.round_num - 1] = 1.0

        # Representation of DB query results (scaled counts)
        kb_count_rep = np.zeros((self.num_slots + 1,)) + db_results_dict['matching_all_constraints'] / 100.
        for key in db_results_dict.keys():
            if key in self.slots_dict:
                kb_count_rep[self.slots_dict[key]] = db_results_dict[key] / 100.

        # Representation of DB query results (binary)
        kb_binary_rep = np.zeros((self.num_slots + 1,)) + np.sum(db_results_dict['matching_all_constraints'] > 0.)
        for key in db_results_dict.keys():
            if key in self.slots_dict:
                kb_binary_rep[self.slots_dict[key]] = np.sum(db_results_dict[key] > 0.)
        # print(kb_binary_rep)

        # represent current slot has value in db result
        db_binary_slot_rep = np.zeros((self.num_slots + 1,))
        db_results = self.db_helper.get_db_results(self.current_informs)
        if db_results:
            # Arbitrarily pick the first value of the dict
            key, data = list(db_results.items())[0]
            # print("size state: {} ".format(self.num_slots + 1))
            # print("first value:   {}".format(data))
            for slot, value in data.items():
                if slot in self.slots_dict and isinstance(value, list) and len(value) > 0:
                    # if slot not in self.current_request_slots:
                    db_binary_slot_rep[self.slots_dict[slot]] = 1.0
        
        # print("-------------------begin element")
        # print(user_act_rep)
        # print(user_inform_slots_rep)
        # print(user_request_slots_rep)
        # print(agent_act_rep)
        # print(agent_inform_slots_rep)
        # print(agent_request_slots_rep)
        # print("-------------------end element")
        # print("---------------------begin hstack")
        list_state = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep]).flatten()
        # print(list_state[:6])
        # print(list_state[12:18])
        # print(list_state[18:30])
        # print(list_state[30:36])
        # print(list_state[36:48])
        # print(list_state[48:60])
        # print("---------------------end hstack")

        # print(np.hstack(
        #     [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
        #      agent_request_slots_rep]))


        state_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep,
             kb_count_rep,db_binary_slot_rep]).flatten()
        # print("---------------------------------------state")
        # print(state_representation)
        # time.sleep(0.5)
        return state_representation

    def update_state_agent(self, agent_action, user_action):
        """
        Updates the dialogue history with the agent's action and augments the agent's action.
        Takes an agent action and updates the history. Also augments the agent_action param with query information and
        any other necessary information.
        Parameters:
            agent_action (dict): The agent action of format dict('intent': string, 'inform_slots': dict,
                                 'request_slots': dict) and changed to dict('intent': '', 'inform_slots': {},
                                 'request_slots': {}, 'round': int, 'speaker': 'Agent')
        """

        if agent_action['intent'] == 'inform':
            assert agent_action['inform_slots']
            # print("intent: inform, current inform_slots: {}".format(self.current_informs))
            # print("current request slot: {}".format(self.current_request_slots))

            inform_slots, list_match_obj = self.db_helper.fill_inform_slot(agent_action['inform_slots'], self.current_informs, user_action)
            agent_action['inform_slots'] = inform_slots
            assert agent_action['inform_slots']
            #########inform slot chỉ có 1 key nên lấy phần tử 0
            key, value = list(agent_action['inform_slots'].items())[0]  # Only one
            assert key != 'match_found'
            assert value != 'PLACEHOLDER', 'KEY: {}'.format(key)

            #######TO DO: Cập nhật lại current inform theo  list_match_obj và value (không chỉ 1 key)(ok)
            # if isinstance(value, tuple):
            #   self.current_informs[key] = list(value)
            # else:
            #   self.current_informs[key] = value
            for key, value in user_action['inform_slots'].items():
                self.current_informs[key] = [value]
            for map_key in list_map_key:
                if map_key != key:
                    self.current_informs[key] = ['']
            list_works_obj = []
            list_name_place_obj = []
            list_address_obj = []
            list_time_obj = []
            for match_obj in list_match_obj:
                list_works_obj.append(match_obj["works"])
                list_name_place_obj.append(match_obj["name_place"])
                list_address_obj.append(match_obj["address"])
                list_time_obj.append(match_obj["time"])
            self.current_informs["works"].append(list_works_obj)
            self.current_informs["name_place"].append(list_name_place_obj)
            self.current_informs["address"].append(list_address_obj)
            self.current_informs["time"].append(list_time_obj)


        # If intent is match_found then fill the action informs with the matches informs (if there is a match)
        elif agent_action['intent'] == 'match_found':
            assert not agent_action['inform_slots'], 'Cannot inform and have intent of match found!'
            # print("intent: match found, current informs: {}".format(self.current_informs))
            list_match_obj = []
            db_results = self.db_helper.get_db_results(self.current_informs)
            if db_results:
                # Arbitrarily pick the first value of the dict


                # list_results = list(db_results.items())
                # index = 0
                # key, data = list_results[index]
                # while index < len(list_results) and isinstance(data[index][self.current_request_slots[0]],list) and len(data[index][self.current_request_slots[0]]) == 0:
                #     key, data = list(db_results.items())[index]
                #     index += 1
                db_results_no_empty = {}
                if self.current_request_slots[0] != usersim_default_key:
                    for key, data in db_results.items():
                        if isinstance(data[self.current_request_slots[0]], list) and len(data[self.current_request_slots[0]]) > 0:
                            db_results_no_empty[key] = copy.deepcopy(data)
                if db_results_no_empty:
                    key, data = list(db_results_no_empty.items())[0]
                    data = list(db_results_no_empty.values())
                    # print("MATCH FOUND: filtered only not empty data ")
                else:
                    key, data = list(db_results.items())[0]
                    data = list(db_results.values())
                # key, data = list(db_results.items())[0]
                ### MATCH found thì chỗ này giữ nguyên
                agent_action['inform_slots'] = {key:copy.deepcopy(data)}
                agent_action['inform_slots'][self.match_key] = str(key)
                #########TO DO : tạo list_match_obj từ first result data dựa vào điều kiện current inform (ok)
                key = agent_action['inform_slots']['activity']
                first_result_data = agent_action['inform_slots'][key][0]
                ###nếu như trong điều kiện có key map thì mới trả về list match obj 
                if "works" in self.current_informs or "name_place" in self.current_informs or "address" in self.current_informs or "time" in self.current_informs:
                    if "time_works_place_address_mapping" in first_result_data and first_result_data["time_works_place_address_mapping"] not in [None,[]]:
                        list_constraint_obj = []
                        len_constraint_map = len(self.current_informs["works"][1])
                        for i in len_constraint_map:
                            list_constraint_obj.append({"works":self.current_informs["works"][1][i],"address":self.current_informs["address"][1][i],"name_place":self.current_informs["name_place"][1][i],"time":self.current_informs["time"][1][i]})
                        if len_constraint_map != 0:
                            list_map_obj = first_result_data["time_works_place_address_mapping"]
                            for map_obj in list_map_obj:
                                match = False
                                for constraint_obj in list_constraint_obj:
                                    if check_match_obj(constraint_obj,map_obj):
                                        match = True
                                        break
                                if match == True:
                                    list_match_obj.append(map_obj)





            else: ####### no match thì giữ nguyên 
                agent_action['inform_slots'][self.match_key] = 'no match available'
            ### giữ nguyên 
            self.current_informs[self.match_key] = agent_action['inform_slots'][self.match_key]

        ######### TO DO : bỏ thêm thông tin match object vào action (ok)
        agent_action.update({'round': self.round_num, 'speaker': 'Agent', 'list_match_obj': list_match_obj})
        
        self.history.append(agent_action)
        print("------------------------------------history in update state agent")
        print(self.history)

    def update_state_user(self, user_action):
        """
        Updates the dialogue history with the user's action and augments the user's action.
        Takes a user action and updates the history. Also augments the user_action param with necessary information.
        Parameters:
            user_action (dict): The user action of format dict('intent': string, 'inform_slots': dict,
                                 'request_slots': dict) and changed to dict('intent': '', 'inform_slots': {},
                                 'request_slots': {}, 'round': int, 'speaker': 'User')
        """
        list_match_obj = []
        if "list_match_obj" in user_action:
            list_match_obj = user_action["list_match_obj"]
        ###########TO DO cập nhật current inform va current request slot dựa vào value và list_match_obj (không chỉ 1 key) (ok)
        if key not in list_map_key:
            for key, value in user_action['inform_slots'].items():
                self.current_informs[key] = value
        else:
            for key, value in user_action['inform_slots'].items():
                self.current_informs[key] = [value]
            for map_key in list_map_key:
                if map_key != key:
                    self.current_informs[key] = ['']
            list_works_obj = []
            list_name_place_obj = []
            list_address_obj = []
            list_time_obj = []
            for match_obj in list_match_obj:
                list_works_obj.append(match_obj["works"])
                list_name_place_obj.append(match_obj["name_place"])
                list_address_obj.append(match_obj["address"])
                list_time_obj.append(match_obj["time"])
            self.current_informs["works"].append(list_works_obj)
            self.current_informs["name_place"].append(list_name_place_obj)
            self.current_informs["address"].append(list_address_obj)
            self.current_informs["time"].append(list_time_obj)
        for key, value in user_action['request_slots'].items():
            if key not in self.current_request_slots:
                self.current_request_slots.append(key)
        user_action.update({'round': self.round_num, 'speaker': 'User'})
        self.history.append(user_action)
        self.round_num += 1
        print("---------------------------------------------history in update state user")
        print(self.history)
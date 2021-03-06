from collections import defaultdict
from dialogue_config import no_query_keys, usersim_default_key
import copy
 
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from flask_cors import CORS
import re
from pymongo import MongoClient

###### tạm thời để đây vì import từ constants báo lỗi no module name constants
list_map_key = ["works", "name_place", "address", "time"]

# app = Flask(__name__)
# CORS(app)
 
# app.config["MONGO_URI"] = "mongodb://caochanhduong:bikhungha1@ds261626.mlab.com:61626/activity?retryWrites=false"
# mongo = PyMongo(app)
 
 
 
 
# client = MongoClient()
# client = MongoClient('mongodb://caochanhduong:bikhungha1@ds261626.mlab.com:61626/activity?retryWrites=false')
# db = client.activity
class DBQuery:
    """Queries the database for the state tracker."""
 
    def __init__(self, database):
        """
        The constructor for DBQuery.
 
        Parameters:
            database (dict): The database in the format dict(long: dict)
        """
 
        self.db = database
        # {frozenset: {string: int}} A dict of dicts
        self.cached_db_slot = defaultdict(dict)
        # {frozenset: {'#': {'slot': 'value'}}} A dict of dicts of dicts, a dict of DB sub-dicts
        self.cached_db = defaultdict(dict)
        self.no_query = no_query_keys
        self.match_key = usersim_default_key
 
    def fill_inform_slot(self, inform_slot_to_fill, current_inform_slots, user_action):
        """
        Given the current informs/constraints fill the informs that need to be filled with values from the database.
 
        Searches through the database to fill the inform slots with PLACEHOLDER with values that work given the current
        constraints of the current episode.
 
        Parameters:
            inform_slot_to_fill (dict): Inform slots to fill with values
            current_inform_slots (dict): Current inform slots with values from the StateTracker
 
        Returns:
            dict: inform_slot_to_fill filled with values
        """
 
        # For this simple system only one inform slot should ever passed in
        assert len(inform_slot_to_fill) == 1
 
        key = list(inform_slot_to_fill.keys())[0]
 
        # This removes the inform we want to fill from the current informs if it is present in the current informs
        # so it can be re-queried
        current_informs = copy.deepcopy(current_inform_slots)
        current_informs.pop(key, None)
 
        # db_results is a dict of dict in the same exact format as the db, it is just a subset of the db
        db_results = self.get_db_results(current_informs, user_action)
 
        # filled_inform = {}
        # values_dict = self._count_slot_values(key, db_results)
        # if values_dict:
        #     # Get key with max value (ie slot value with highest count of available results)
        #     filled_inform[key] = list(max(values_dict, key=values_dict.get))
        # else:
        #     filled_inform[key] = 'no match available'
 
        # return filled_inform

        db_results_no_empty = {}
        
        if key != usersim_default_key:
            for i, data in db_results.items():
                list_value = data[key]
                ######do bây giờ chỉ inform thông tin chung nên không cần, thông tin riêng đã chứa trong list match obj
                # if key in list_map_key and "time_works_place_address_mapping" in data and isinstance(data["time_works_place_address_mapping"],list):
                #     list_obj_map = data["time_works_place_address_mapping"]
                #     for obj_map in list_obj_map:
                #         if key in obj_map:
                #             list_value.append(obj_map[key])
                # list_value = list(set(list_value))
                if isinstance(list_value, list) and len(list_value) > 0:
                    db_results_no_empty[i] = copy.deepcopy(data)
      
        filled_inform = {}
        values_dict = {}
        list_match_obj = []
        db_search_map = {}
        if db_results_no_empty:
            db_search_map = db_results_no_empty
            #chỗ này count slot values kiểu gì? có thể count như cũ cũng được
            values_dict = self._count_slot_values(key, db_results_no_empty)
            # print("INFORM: filtered out, values_dict: {}".format(values_dict))
        else:
            db_search_map = db_results
            values_dict = self._count_slot_values(key, db_results)
            # print("INFORM: can not filtered out, values_dict: {}".format(values_dict))
        
        ## Giữ nguyễn inform kết quả đầu tiên nếu là activity
        if key == usersim_default_key:
            filled_inform[key] = list(db_results)[0]

        elif values_dict:
            # Get key with max value (ie slot value with highest count of available results)
            filled_inform[key] = list(max(values_dict, key=values_dict.get))
            # Lấy các object công việc con giữ nguyên 
            if filled_inform[key] != [] and key in list_map_key:
                for i, data in db_search_map.items():
                    list_value = data[key]
                    ######do bây giờ chỉ inform thông tin chung nên không cần, thông tin riêng đã chứa trong list match obj
                    # if "time_works_place_address_mapping" in data and isinstance(data["time_works_place_address_mapping"],list):
                    #     list_obj_map = data["time_works_place_address_mapping"]
                    #     for obj_map in list_obj_map:
                    #         if key in obj_map:
                    #             list_value.append(obj_map[key])
                    # list_value = list(set(list_value))
                    # Tìm được data đầu tiên thỏa, sẽ break sau khi so và lọc object
                    # cần so kiểu đặc biệt
                    if list_value == filled_inform[key]:
                        ##TO DO : dựa vào cấu trúc current inform đặc biệt (ok)
                        ## để lấy các object và trả về 
                        if "works" in current_informs or "name_place" in current_informs or "address" in current_informs or "time" in current_informs:
                            if "time_works_place_address_mapping" in data and data["time_works_place_address_mapping"] not in [None,[]]:
                                list_constraint_obj = []
                                len_constraint_map = len(current_informs["works"][1])
                                for i in len_constraint_map:
                                    list_constraint_obj.append({"works":current_informs["works"][1][i],"address":current_informs["address"][1][i],"name_place":current_informs["name_place"][1][i],"time":current_informs["time"][1][i]})
                                if len_constraint_map != 0:
                                    list_map_obj = data["time_works_place_address_mapping"]
                                    for map_obj in list_map_obj:
                                        match = False
                                        for constraint_obj in list_constraint_obj:
                                            if check_match_obj(constraint_obj,map_obj):
                                                match = True
                                                break
                                        if match == True:
                                            list_match_obj.append(map_obj)
                        break

        #nếu không tìm được kq nào thì giữ nguyên inform no match    
        else:
            filled_inform[key] = 'no match available'

        return filled_inform, list_match_obj
 
    def _count_slot_values(self, key, db_subdict):
        ########## không cần sửa do không đụng đến current inform, chỉ có đêm value theo key 
        """
        Return a dict of the different values and occurrences of each, given a key, from a sub-dict of database
 
        Parameters:
            key (string): The key to be counted
            db_subdict (dict): A sub-dict of the database
 
        Returns:
            dict: The values and their occurrences given the key
        """
 
        # slot_values = defaultdict(int)  # init to 0
        # for id in db_subdict.keys():
        #     current_option_dict = db_subdict[id]
        #     # If there is a match
        #     if key in current_option_dict.keys():
        #         slot_value = current_option_dict[key]
        #         tp_slot_value = tuple(slot_value)
        #         # This will add 1 to 0 if this is the first time this value has been encountered, or it will add 1
        #         # to whatever was already in there
        #         slot_values[tp_slot_value] += 1
        # return slot_values

        slot_values = defaultdict(int)  # init to 0
        # print(slot_values)
        for id in db_subdict.keys():
            current_option_dict = db_subdict[id]
            # If there is a match
            if key in current_option_dict.keys():
                slot_value = current_option_dict[key]
                if key in list_map_key and "time_works_place_address_mapping" in current_option_dict and isinstance(current_option_dict["time_works_place_address_mapping"],list):
                    list_obj_map = current_option_dict["time_works_place_address_mapping"]
                    for obj_map in list_obj_map:
                        if key in obj_map:
                            slot_value.append(obj_map[key])
                # đồng bộ các value để không cần xét đến thứ tự xuất hiện
                slot_value = list(set(slot_value))

                # print(slot_value)
                if any(isinstance(i,list) for i in slot_value):
                  slot_value = [value for sub_list in slot_value for value in sub_list]

                tp_slot_value = tuple(slot_value)
                # print(type(tp_slot_value))
                # This will add 1 to 0 if this is the first time this value has been encountered, or it will add 1
                # to whatever was already in there
                slot_values[tp_slot_value] += 1
        return slot_values

 
    def get_db_results(self, constraints, user_action):
        ####### không cần sửa
        """
        Get all items in the database that fit the current constraints.
 
        Looks at each item in the database and if its slots contain all constraints and their values match then the item
        is added to the return dict.
 
        Parameters:
            constraints (dict): The current informs
 
        Returns:
            dict: The available items in the database
        """
 
        # Filter non-queryable items and keys with the value 'anything' since those are inconsequential to the constraints
        new_constraints = {k: v for k, v in constraints.items() if k not in self.no_query and v is not 'anything'}
 
        tuple_new_constraint=copy.deepcopy(new_constraints)
        # print(tuple_new_constraint)
        inform_items ={k:tuple(v) for k,v in tuple_new_constraint.items()}.items()
        inform_items = frozenset(inform_items)
 
        # inform_items = frozenset(new_constraints.items())
        cache_return = self.cached_db[inform_items]
 
        if cache_return == None:
            # If it is none then no matches fit with the constraints so return an empty dict
            return {}
        # if it isnt empty then return what it is
        if cache_return:
            return cache_return
        # else continue on
 
        available_options = {}
        regex_constraint = self.convert_to_regex_constraint(new_constraints, user_action)
        results = self.db.activities.find(regex_constraint)
        for result in results:
            #đổi từ object id sang string và dùng id đó làm key (thay vì dùng index của mảng để làm key vì không xác định đc index)
            result["_id"] = str(result["_id"])
            available_options.update({result['_id']:result})
            self.cached_db[inform_items].update({result['_id']: result})
 
 
 
        # for id in self.database.keys():
        #     current_option_dict = self.database[id]
        #     # First check if that database item actually contains the inform keys
        #     # Note: this assumes that if a constraint is not found in the db item then that item is not a match
        #     if len(set(new_constraints.keys()) - set(self.database[id].keys())) == 0:
        #         match = True
        #         # Now check all the constraint values against the db values and if there is a mismatch don't store
        #         for k, v in new_constraints.items():
        #             if str(v).lower() != str(current_option_dict[k]).lower():
        #                 match = False
        #         if match:
        #             # Update cache
        #             self.cached_db[inform_items].update({id: current_option_dict})
        #             available_options.update({id: current_option_dict})
 
        # if nothing available then set the set of constraint items to none in cache
        if not available_options:
            self.cached_db[inform_items] = None
 
        return available_options
    def get_db_results_for_slots(self, current_informs,user_action):
        """
        Counts occurrences of each current inform slot (key and value) in the database items.
 
        For each item in the database and each current inform slot if that slot is in the database item (matches key
        and value) then increment the count for that key by 1.
 
        Parameters:
            current_informs (dict): The current informs/constraints
 
        Returns:
            dict: Each key in current_informs with the count of the number of matches for that key
        """
        ##########TO DO : dựa vào user action, current inform đếm số match cho từng key (chia 2 TH user request và không request) 
        # The items (key, value) of the current informs are used as a key to the cached_db_slot
        # print()
        # print(type(self.cached_db_slot))
        tuple_current_informs=copy.deepcopy(current_informs)
        # print(tuple_current_informs)
        inform_items ={k:tuple(v) for k,v in tuple_current_informs.items()}.items()
        inform_items = frozenset(inform_items)
        # # A dict of the inform keys and their counts as stored (or not stored) in the cached_db_slot
        cache_return = self.cached_db_slot[inform_items]
        temp_current_informs=copy.deepcopy(current_informs)
        if cache_return:
            return cache_return
 
        # If it made it down here then a new query was made and it must add it to cached_db_slot and return it
        # Init all key values with 0
        db_results = {key: 0 for key in current_informs.keys()}
        db_results['matching_all_constraints'] = 0
 
        # for id in self.database.keys():
        # all_slots_match = True
        for CI_key, CI_value in current_informs.items():
        # Skip if a no query item and all_slots_match stays true
            if CI_key in self.no_query:
                continue
            # If anything all_slots_match stays true AND the specific key slot gets a +1
            if CI_value == 'anything':
                db_results[CI_key] = self.db.activities.count()
                del temp_current_informs[CI_key]
                continue
            db_results[CI_key]=self.db.activities.count(self.convert_to_regex_constraint({CI_key:CI_value},user_action))
            # print(CI_key)
            # print(db_results[CI_key])
           
        # current_informs_constraint={k:v.lower() for k,v in temp_current_informs.items()}
        db_results['matching_all_constraints'] = self.db.activities.count(self.convert_to_regex_constraint(temp_current_informs,user_action))
       
        # update cache (set the empty dict)
        self.cached_db_slot[inform_items].update(db_results)
        assert self.cached_db_slot[inform_items] == db_results
        return db_results
    # def convert_to_regex_constraint(self, constraints):
    #     list_and_out = []
    #     list_and_in = []
    #     ele_match_obj = {}
    #     list_or = []
    #     for k,values in constraints.items():
    #         list_pat = []
    #         for value in values:
    #             list_pat.append(re.compile(".*{0}.*".format(value)))
    #         # regex_constraint_dict[k] = {"$all":list_pat}
    #         if k not in ["works","name_place","address","time"]:
    #             list_and_out.append({k:{"$all":list_pat}})
    #         else:
    #             list_and_in.append({k:{"$all":list_pat}})
    #             ele_match_obj[k] = {"$all":list_pat}
    #     if list_and_in != [] :
    #         list_or = [{"$and":list_and_in},{"time_works_place_address_mapping":{"$all":[{"$elemMatch":ele_match_obj}]}}]
    #         list_and_out.append({"$or" : list_or})
    #     regex_constraint_dict = {"$and":list_and_out}
    #     return regex_constraint_dict

    def convert_to_regex_constraint(self, constraints, user_action):
        list_and_out = []
        list_and_in = []
        ele_match_obj = {}
        list_or = []
        regex_constraint_dict = {}
        if user_action["intent"] == "request":
            ##### TO DO: query theo cặp 1 là ngoài 2 là trong (dựa vào constaints (tức current inform))
        else:
            ##### TO DO: query theo cấu trúc dữ liệu đặc biệt, lấy từ
            ### constraint  (dựa vào constaints (tức current inform))

            
        # for k,values in constraints.items():
        #     if k not in list_map_key:
        #         list_pattern = []
        #         for value in values:
        #             list_pattern.append(re.compile(".*{0}.*".format(value)))
        #         if list_pattern != []:
        #             list_and_out.append({k: {"$all": list_pattern}})
        #     else:
        #         for value in values:
        #             list_and_in.append({
        #                 "$or" : [
        #                             {
        #                                 k: {
        #                                     "$all": [re.compile(".*{0}.*".format(value))]
        #                                 }
        #                             },
        #                             {    
        #                                 "time_works_place_address_mapping": {
        #                                     "$all": [
        #                                                 {
        #                                                     "$elemMatch": {
        #                                                             k: {
        #                                                                 "$all": [re.compile(".*{0}.*".format(value))]
        #                                                             }
        #                                                     }
        #                                                 }
        #                                             ]
        #                                 }
        #                             }
        #                         ]
        #             })

        # if list_and_in != []:
        #     list_and_out.append({"$and": list_and_in})

        # if list_and_out != []:
        #     regex_constraint_dict = {"$and":list_and_out}
        return regex_constraint_dict

                


from pymongo import MongoClient
client = MongoClient('mongodb://caochanhduong:bikhungha1@ds261626.mlab.com:61626/activity?retryWrites=false')
database = client.activity
dbquery = DBQuery(database)
# print(dbquery.get_db_results({"name_activity":["đêm vui tất niên tết ấm áp","c"],"time":["10h","15/01/19"],"works": ["ức"]}))
# print(dbquery.convert_to_regex_constraint({"name_activity":["đêm vui tất niên tết ấm áp"],"time":["10h","15/01/20"]}))
print(dbquery._count_slot_values("time",{}))
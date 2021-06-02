import firebase_admin
from firebase_admin import db,credentials,auth
import json
import getpass
from PyQt5.QtWidgets import  QInputDialog, QLineEdit, QApplication
from PyQt5.QtCore import QCoreApplication
import time
import pyrebase


#password prompt dialog box 
def prompt_password(user):
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication([])
    text, ok = QInputDialog.getText(
        None,
        "Credential",
        "user {}:".format(user),
        QLineEdit.Password)
    if ok and text:
        return text
    raise ValueError("Must specify a valid password")

# fetching existing user 
def existing_user():
    email=input("Email Id:")
    password=prompt_password(email)
    auth=firebase.auth()
    signin=auth.sign_in_with_email_and_password(email,password)
    user_uid=signin['idToken']
    user_detail=auth.get_account_info(user_uid)
    userid=user_detail['users']
    local_id=userid[0]['localId']
    user_uid=local_id
    #simulating real_time   
    #----------------for existing user--------------
    existing_data=db.reference("/").child(user_id).get()
    # print(existing_data)
    # fetching and storing child node values
    existing_blinks={}
    existing_blinks=existing_data['blinks']
    existing_yawns=existing_data['yawns']
    existing_drowsiness=existing_data['drowsiness']
    total_blink=existing_data['total_blink']
    total_yawns=existing_data['total_yawns']
    total_drowsiness=existing_data['total_drowsiness']
    trips=existing_data['trips']
    
    # counter value for stimulation
    blink_counter=5
    yawn_counter=5
    drowsiness_counter=1
    trip_counter=1
    
    #updating new values in database
    total_blink+=blink_counter
    total_yawns+=yawn_counter
    total_drowsiness+=drowsiness_counter
    trips+=trip_counter
    
    count=5
    for i in range(0,count):
        #adding keys as time stamps
        #adding values as yawn blinks drowsiness
        time_stamp=int(time.time())
        time.sleep(1)
        existing_yawns[time_stamp]=i
        existing_blinks[time_stamp]=i
        existing_drowsiness[time_stamp]=i

    data={"yawns":existing_yawns,
        "blinks":existing_blinks,
        "drowsiness":existing_drowsiness,
        "total_yawns":total_yawns,
        "total_drowsiness":total_drowsiness,
        "total_blink":total_blink,
        "trips":trips}
    update_data = db.reference("/").child(user_id)
    red=update_data.update(data)
    
    return local_id #return uid of the signed user

#adding new user 
def create_user():
    email=input("Email Id:")
    password=prompt_password(email)
    user=auth.create_user(email=email,password=password)
    print("User Added successfully ")
    user_id=user.uid
       # Initializing dummy variables
    yawn={}
    blink={}
    drowsiness={}
    val=1
    yawn["a"]=val
    blink["a"]=val
    drowsiness["a"]=val
    current_total_yawns=0 #maximum yawn in single program run
    current_total_blinks=0 #maximum yawn in single program run
    current_total_drowsiness=0 #maximum yawn in single program run(a.k.a "Single Critical Alert")
    trips=0
    #Storing root structure in the database
    data={"yawns":yawn,
        "blinks":blink,
        "drowsiness":drowsiness,
        "total_yawns":current_total_yawns,
        "total_drowsiness":current_total_drowsiness,
        "total_blink":current_total_blinks,
        "trips":trips}
    ref = db.reference("/").child(user_id)
    red=ref.set(data)
    return user_id #return uid of the signed user

if __name__=="__main__":
    config=  {
        
    "apiKey": "AIzaSyC_9MN-e2kLYGoXEa-ujZhMJ5KEDhxrxv0",
    "authDomain": "drowsi-6f166.firebaseapp.com",
    "databaseURL": "https://drowsi-6f166-default-rtdb.firebaseio.com",
    "projectId": "drowsi-6f166",
    "storageBucket": "drowsi-6f166.appspot.com",
    "messagingSenderId": "198177358258",
    "appId": "1:198177358258:web:220473766070b40a1224ab",
    "measurementId": "G-FD48W8GKN9"
    }
    
    if not firebase_admin._apps:
       cred_obj = firebase_admin.credentials.Certificate('E:\MrDrowsi\Drowsiness_Detection-master\serviceAccountKey.json')
       default_app = firebase_admin.initialize_app(cred_obj, {
	'databaseURL': 'https://drowsi-6f166-default-rtdb.firebaseio.com/'})
       firebase=pyrebase.initialize_app(config)
    
    user_type=int(input("\n1.New User \n2.Existing User\n"))
    if user_type==1:
        user_id=create_user()
    else:
        user_id=existing_user()


    

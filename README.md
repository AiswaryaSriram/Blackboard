# Blackboard
An e-learning website for students to view educational videos, course materials and upload assignments.

1. Make sure the flask server (server.py) is running. This file should be located within src.
2. All API calls are within server.py
3. server.py also has an API for automatic essay grading. This uses files within src (my_model_w2v.h5)
4. In order to regenerate the LSTM word2vec model for training the automated essay grading machine use: project.py (inside src)
5. Html scripts are located within the Blackboard folder.
6. Dashbpard page is our_dashboard.html
7. CSS and Bootstrap dependencies are within material-dashboard folder.
8. The Blackboard folder must be placed in /opt/lampp/htdocs.
9. The sql commands for creating the 4 tables are present in sqlcommands.txt
10. Within wtproject2 extract material-dashboard.zip

Steps to get the website up and running:

1. Run server.py
2. Start Apache
3. Create Tables in database using (sqlcommands.txt)
4. Open our_dashboard.html on localhost.


Dependencies:
1. Python Version 3
2. Anaconda (Or individually install keras, numpy, sklearn etc)
3. npm
4. Apache
5. MySql
6. Flask Dependencies

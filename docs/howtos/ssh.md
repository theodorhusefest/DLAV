# HowTo SSH

Explanations for working remotely through SSH.

## Command Line

Connect to SSH via shell:

    ssh your_name@3.15.217.164 -p 2211
    
Copy files via Secure Copy (scp):

    scp <file> <username>@<IP address or hostname>:<Destination>
    
Explained here:

    https://help.ubuntu.com/community/SSH/TransferFiles


## FileZilla

Download & install:

    https://filezilla-project.org/
    
Setup to connect with Kojis AWS:

Tap on "Site Manager" (icon top left), 
tap "New Site", for "Protocol" choose "SSH" 
and enter the following details:

    Host: 3.15.217.164
    Port: 2211
    Logon Type: Normal
    User: your_name
    Passwort: your_passwort

## Remote Jupyter Notebook

Good explanation here:

    https://amber-md.github.io/pytraj/latest/tutorials/remote_jupyter_notebook


## tmux

Why use: close SSH terminals while keeping tasks (e.g. training alive).

First, connect through SSH via terminal.

###Check tmux sessions currently running

    tmux ls
    
###Create new tmux session

    tmux new -s name_of_your_session
    
After that you are in a new tmux session.

###Enter existing tmux session

    tmux a -t name_of_your_session

##Leave session

Press key combination

    CTRL + b
    
Then press 

    d
    
You can then close your SSH connection.

###Delete a session

    tmux kill-session -t name_of_your_session
    

    
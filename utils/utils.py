import smtplib
from email.mime.text import MIMEText

def send_mail(sendto, msg_header, msg_body, user, passwd):
    smtpsrv = "smtp.office365.com"
    smtpserver = smtplib.SMTP(smtpsrv, 587) 

    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.login(user, passwd)

    msg = MIMEText(msg_body)
    msg['From'] = user
    msg['To'] = sendto
    msg['Subject'] = msg_header

    smtpserver.sendmail(user, sendto, msg.as_string())
    smtpserver.close()
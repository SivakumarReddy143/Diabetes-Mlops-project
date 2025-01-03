from diabetes.logging.logger import logging
import sys

class DiabetesException(Exception):
    def __init__(self,error_message,error_details:sys):
        self.error_message=error_message
        _,_,exc_tb=error_details.exc_info()
        self.lineno=exc_tb.tb_lineno
        self.filename=exc_tb.tb_frame.f_code.co_filename
    
    def __str__(self):
        return "Error occured in python script [{0}] in line number [{1}] error message [{2}]".format(
            self.filename,self.lineno,self.error_message
        )

if __name__=="__main__":
    try:
        print(1/0)
    except Exception as e:
        raise DiabetesException(e,sys)
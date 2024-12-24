import sys
import logging

def err_msg_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message= f'error occured in python script {file_name} with line number {exc_tb.tb_lineno} and the error message is {error}'

    return error_message

class Custom_exception(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message= err_msg_detail(error_message,error_detail=error_detail)
    def __str__(self):
        return self.error_message
    


if __name__ == '__main__':
    try:
        a = 1/0
    except Exception as e:
        logging.info('dividing by zero')
        raise Custom_exception(e,sys)
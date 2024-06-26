'''
custom_logger.py
로깅 세팅이 귀찮아서 후딱 만듬
채널 핸들러 만들어 두긴 했는데 어차피 안쓰잖아? 한잔해 

작성자: 윤성진
'''
import logging

def get_logger(name = 'my_logger' ):
    # 로거 설정
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 콘솔 핸들러와 포매터 설정
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # 파일 핸들러와 포매터 설정 (선택 사항)
    fh = logging.FileHandler(f'{name}.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # 로거에 핸들러 추가
    #logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info('START_LOGGING========================================================')
    return logger
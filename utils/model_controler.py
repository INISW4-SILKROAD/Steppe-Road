'''
model_controler.py
범용 모델 컨트롤 모듈
학습, 프리징, score 지원 
"코드가 너무 길면 다른 대로 빼서 코드 길이를 줄여라" - 윤성진 
내가 짰는데 너무 길어서 차마 손도 못대겠다. 이게 인생인가

작성자: 윤성진
'''
import pickle

import torch, os
from tqdm import tqdm

def freeze_model(model, parent= 'model', / , exception = []):
    '''
    모델의 파라미터 중 exception에 포함된 레이어를 제외하고 requires_grad를 False로 바꾸어 줌
    parent는 재귀를 위한 파라미터이니 건들지 말 것
    '''
    for name, child in model.named_children():
        if name in exception:
            for n, param in child.named_parameters():
                param.requires_grad = True
                print(parent, name, n, param.requires_grad)
                continue

        else:
            for param in child.parameters():
                param.requires_grad = False
            freeze_model(child,name, exception)            

def run_train_test_1_input(
    model, train_loader, test_loader, 
    criterion, optimizer, best_test_loss,
    num_epochs, test_step, 
    logger, device, path):
    '''
    입력이 1개인 학습루프입니다. 
    '''

    train_losses = []
    test_losses = []

    # 기존 가중치 있으면 불러옴
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))

    # 학습 루프
    for epoch in range(num_epochs):
        # train loss 계산
        model.train()
        running_loss = 0.0
        batch = 0
        
        # 배치 학습 진행
        for data, label in train_loader:
            # 학습모드 변경 
            batch += 1
            optimizer.zero_grad()

            # 데이터 로드 및 gpu 전송
            data, label = data.to(device), label.to(device)
            logger.info(f'train batch {batch} start')

            # 모델 forward
            outputs = model(data)
            logger.debug(f'train batch {batch} output : {outputs.shape} ')
            
            # 배치 로스 계산
            loss = criterion(outputs, data)        
            logger.debug(f'train batch {batch} loss : {loss:.4f} ')

            # 가중치 업데이트
            loss.backward()
            optimizer.step()

            # 배치 로스 가중
            running_loss += loss.item()
            logger.info(f'train batch {batch} end, running_loss: {running_loss:.4f}')

        # train 로스 계산
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        logger.info(f'Epoch {epoch}/{num_epochs - 1}, Loss: {train_loss:.4f}')

        # 스텝마다 검증
        if epoch % test_step == 0:
            # test 로스 계산
            model.eval()
            test_loss = 0.0
            batch = 0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for data, label in test_loader:
                    batch +=1
                    
                    # 데이터 로드
                    data, label = data.to(device), label.to(device)
                    logger.info(f'valid batch {batch} start')
                    
                    # forward
                    outputs = model(data)
                    logger.debug(f'valid batch {batch} output : {outputs.shape}')

                    # batch loss 계산
                    loss = criterion(outputs, label)
                    logger.debug(f'valid batch {batch} loss : {loss:.4f}')
                    
                    # batch loss 가중
                    test_loss += loss.item()
                    logger.info(f'valid batch {batch} end, running_loss: {test_loss:.4f}')

                    # scoring을 위해 결과 저장
                    all_preds.append(outputs.cpu())
                    all_targets.append(label.cpu())
                    
            # test loss 최솟값 나오면 가중치 저장
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), path)
                logger.info(f'UPDATE best_test_loss :{best_test_loss:.4f}')


            score = calculate_accuracy(all_preds, all_targets, logger)
            test_loss = test_loss / len(test_loader.dataset)
            logger.info(f'TOTAL RUN {epoch}/{num_epochs - 1}, Train_Loss: {train_loss:.4f},  Test_Loss: {test_loss:.4f}')        
            logger.info(f'Score: {score:.4f}')
    
    return best_test_loss, train_losses, test_losses

def run_train_test_2_input(
    model, train_loader, test_loader, 
    criterion, optimizer, best_test_loss, 
    num_epochs, test_step, 
    logger, device, path
    ):
    '''
    입력이 2개인 모델의 학습루프입니다. test_step마다 test를 진행합니다.  
    '''

    train_losses = []
    test_losses = []

    if os.path.exists(path):
        model.load_state_dict(torch.load(path))

    for epoch in range(num_epochs):
        # train loss 계산
        train_loss = _train_model_2(
                model = model, 
                train_loader=train_loader, 
                criterion=criterion, 
                optimizer=optimizer, 
                logger=logger,
                device=device)
        train_losses.append(train_loss)
        logger.info(f'Epoch {epoch}/{num_epochs - 1}, Loss: {train_loss:.4f}')

        # step 마다 test 실행
        if epoch % test_step == 0:    
            test_loss, score = _test_model_2(
                    model=model, 
                    test_loader=test_loader, 
                    criterion=criterion, 
                    logger=logger,
                    device=device)
            test_losses.append(test_loss)

            # valid_loss 최솟값 저장
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), path)
                logger.info(f'UPDATE best_test_loss :{best_test_loss:.4f}')

            logger.info(f'TOTAL RUN {epoch}/{num_epochs - 1}, Train_Loss: {train_loss:.4f},  Valid_Loss: {test_loss:.4f}')
            logger.info(f'score: {score}')
            
    return best_test_loss, train_losses, test_losses

def _train_model_2(model, train_loader, criterion, optimizer, logger, device):
    model.train()
    running_loss = 0.0
    batch = 0
    
    for image_em, portion, labels in tqdm(train_loader):
        batch += 1
        optimizer.zero_grad()
        logger.info(f'train batch {batch} start')
        
        # forward
        image_em, portion ,labels = image_em.to(device), portion.to(device), labels.to(device)
        outputs = model(image_em, portion)
        logger.debug(f'train batch {batch} output : {outputs.shape} ')

        # batch loss 계산
        loss = criterion(outputs, labels)
        logger.debug(f'train batch {batch} loss : {loss:.4f} ')
        
        # 가중치 업데이트
        loss.backward()
        optimizer.step()
        
        # batch loss 가중
        running_loss += loss.item() * image_em.size(0)
        logger.info(f'train batch {batch} end, running_loss: {running_loss:.4f}')
    
    # train loss 계산
    train_loss = running_loss / len(train_loader.dataset)
    return train_loss

def _test_model_2(model, test_loader, criterion, logger, device):
    model.eval()
    test_loss = 0.0
    batch = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for image_em, portion, labels in test_loader:
            batch +=1
            logger.info(f'valid batch {batch} start')
            
            # forward
            image_em, portion ,labels = image_em.to(device), portion.to(device), labels.to(device)
            outputs = model(image_em, portion)
            logger.debug(f'valid batch {batch} output : {outputs.shape}')
            
            # batch loss 계산
            loss = criterion(outputs, labels)
            logger.debug(f'valid batch {batch} loss : {loss:.4f}')
            
            # batch loss 가중
            test_loss += loss.item() * image_em.size(0)
            logger.info(f'valid batch {batch} end, running_loss: {test_loss:.4f}')

            all_preds.append(outputs.cpu())
            all_targets.append(labels.cpu())

    # train loss 및 score 계산
    score = calculate_accuracy(all_preds, all_targets, logger)
    test_loss = test_loss / len(test_loader.dataset)
    
    return test_loss, score

def calculate_accuracy(all_preds, all_targets):
    '''
    정확도를 계산합니다.
    '''
    preds = torch.cat(all_preds)
    labels = torch.cat(all_targets)

    total_count = [(p - l).tolist() for p, l in zip(preds.max(dim=1)[1],labels)]
    correct = 0
    wrong = 0

    for i in total_count:
        if i == 0. : correct += 1
        else       : wrong   += 1

    score= correct / (correct+wrong)

    return score

def save_result(results, path):
    best_test_loss, train_losses, test_losses = [], [], []
    for i in results:
        best_test_loss.append(i[0])
        train_losses += i[1]
        test_losses += i[2]
    with open(path, 'wb') as f:
        pickle.dump(
            {
                'best_test_loss': best_test_loss,
                'train_losses': train_losses,
                'test_losses': test_losses
            }, f)
        
    return best_test_loss, train_losses, test_losses
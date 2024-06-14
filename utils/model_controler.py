import torch , os
from tqdm import tqdm

def freeze_model(model, parent= '', / , exception = []):
    '''
    모델의 파라미터 중 exception에 포함된 레이어를 제외하고 requires_grad를 False로 바꾸어 줌
    parent는 재귀를 위한 파라미터이니 건들지 말것
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

    if os.path.exists(path):
        model.load_state_dict(torch.load(path))

    for epoch in range(num_epochs):
        # train loss 계산
        model.train()
        running_loss = 0.0
        batch = 0
        for data, label in train_loader:
            batch += 1
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            logger.info(f'train batch {batch} start')


            model.train()
            outputs = model(data)
            logger.debug(f'train batch {batch} output : {outputs.shape} ')
            
            loss = criterion(outputs, data)        
            logger.debug(f'train batch {batch} loss : {loss:.4f} ')

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            logger.info(f'train batch {batch} end, running_loss: {running_loss:.4f}')

        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        logger.info(f'Epoch {epoch}/{num_epochs - 1}, Loss: {train_loss:.4f}')

        if epoch % test_step == 0:
            
            model.eval()
            test_loss = 0.0
            batch = 0
            all_preds = []
            all_targets = []
            touch = ['softness', 'smoothness', 'thickness', 'flexibility']

            with torch.no_grad():
                for data, label in test_loader:
                    batch +=1
                    data, label = data.to(device), label.to(device)

                    logger.info(f'valid batch {batch} start')

                    outputs = model(data)
                    logger.debug(f'valid batch {batch} output : {outputs.shape}')

                    loss = criterion(outputs, label)
                    logger.debug(f'valid batch {batch} loss : {loss:.4f}')
                    
                    test_loss += loss.item()
                    logger.info(f'valid batch {batch} end, running_loss: {test_loss:.4f}')

                    all_preds.append(outputs.cpu())
                    all_targets.append(label.cpu())
            # valid_loss 최솟값 저장
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), path)
                logger.info(f'UPDATE best_test_loss :{best_test_loss:.4f}')

            logger.info(f'TOTAL RUN {epoch}/{num_epochs - 1}, Train_Loss: {train_loss:.4f},  Valid_Loss: {test_loss:.4f}')

            score = calculate_accuracy(all_preds, all_targets, logger)
            test_loss = test_loss / len(test_loader.dataset)
            logger.info(f'Validation Loss: {test_loss:.4f}')
    
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
            logger.info(f'{score}')
            
    return best_test_loss, train_losses, test_losses

def _train_model_2(model, train_loader, criterion, optimizer, logger, device):
    model.train()
    running_loss = 0.0
    batch = 0
    
    for image_em, portion, labels in tqdm(train_loader):
        batch += 1
        optimizer.zero_grad()
        logger.info(f'train batch {batch} start')
        
        image_em, portion ,labels = image_em.to(device), portion.to(device), labels.to(device)
        outputs = model(image_em, portion)
        logger.debug(f'train batch {batch} output : {outputs.shape} ')

        loss = criterion(outputs, labels)
        logger.debug(f'train batch {batch} loss : {loss:.4f} ')

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * image_em.size(0)
        logger.info(f'train batch {batch} end, running_loss: {running_loss:.4f}')
        
    train_loss = running_loss / len(train_loader.dataset)
    return train_loss

def _test_model_2(model, test_loader, criterion, logger, device):
    model.eval()
    test_loss = 0.0
    batch = 0
    all_preds = []
    all_targets = []
    touch = ['softness', 'smoothness', 'thickness', 'flexibility']

    with torch.no_grad():
        for image_em, portion, labels in test_loader:
            batch +=1
            logger.info(f'valid batch {batch} start')

            image_em, portion ,labels = image_em.to(device), portion.to(device), labels.to(device)
            outputs = model(image_em, portion)
            logger.debug(f'valid batch {batch} output : {outputs.shape}')

            loss = criterion(outputs, labels)
            logger.debug(f'valid batch {batch} loss : {loss:.4f}')
            
            test_loss += loss.item() * image_em.size(0)
            logger.info(f'valid batch {batch} end, running_loss: {test_loss:.4f}')

            all_preds.append(outputs.cpu())
            all_targets.append(labels.cpu())

    score = calculate_accuracy(all_preds, all_targets, logger)
    test_loss = test_loss / len(test_loader.dataset)
    logger.info(f'Validation Loss: {test_loss:.4f}')
    
    return test_loss, score

def calculate_accuracy(all_preds, all_targets, logger):
    '''
    정확도를 계산합니다. 
    '''
    preds = torch.cat(all_preds)
    labels = torch.cat(all_targets)

    TOUCH = ['softness', 'smoothness', 'thickness', 'flexibility']
    total_count = [[(p[i].round() - l[i]).tolist() for p, l in zip(preds,labels)] for i in range(len(TOUCH))]
    score = {}

    for index, touch in enumerate(total_count):
        correct = 0
        wrong = 0

        for i in touch: 
            if i == 0. : correct += 1
            else       : wrong   += 1

        score[TOUCH[index]] = correct / (correct+wrong)
    logger.info(f'score: {score}')
    return score

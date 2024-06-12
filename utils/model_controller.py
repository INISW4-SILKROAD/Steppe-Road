import torch , os
from tqdm import tqdm

def freeze_model(model, parent= '',* , exception = []):
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


def train_model(model, train_loader, criterion, optimizer, logger, device, num_epochs=25):
    model.train()
    train_loss = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        batch = 0
        for image_em, portion, labels in tqdm(train_loader, desc = f'batch: {batch+1}/ epoch: {epoch}'):
            batch += 1
            logger.info(f'train batch {batch} start')
            
            image_em, portion ,labels = image_em.to(device), portion.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(image_em, portion)
            logger.info(f'train batch {batch} output : {outputs.shape} ')

            loss = criterion(outputs, labels)
            logger.info(f'train batch {batch} loss : {loss:.4f} ')

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * image_em.size(0)
            logger.info(f'train batch {batch} end, running_loss: {running_loss:.4f}')
            

        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss = epoch_loss
        logger.info(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
   
    return train_loss

# 검증 함수
def test_model(model, test_loader, criterion, logger, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        batch = 0
        for image_em, portion, labels in test_loader:
            batch +=1
            logger.info(f'valid batch {batch} start')

            image_em, portion ,labels = image_em.to(device), portion.to(device), labels.to(device)
           
            outputs = model(image_em, portion)
            logger.info(f'valid batch {batch} output : {outputs.shape} ')

            loss = criterion(outputs, labels)
            logger.info(f'valid batch {batch} loss : {loss:.4f} ')
            
            test_loss += loss.item() * image_em.size(0)
            logger.info(f'valid batch {batch} end, running_loss: {test_loss:.4f}')
    
    test_loss /= len(test_loader.dataset)
    logger.info(f'Validation Loss: {test_loss:.4f}')
    
    return test_loss

def run_train_test_2_input(model, train_loader, test_loader, best_test_loss, criterion, optimizer, num_epochs, test_step, logger, device, path):
    train_loss = []
    test_loss = []
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))

    for epoch in range(num_epochs):
        # train, valid loss 계산
        train_loss.append(
            train_model(
                model = model, 
                train_loader=train_loader, 
                criterion=criterion, 
                optimizer=optimizer, 
                logger=logger,
                device=device, 
                num_epochs=test_step)
            )
            
        test_loss.append(
            test_model(
                model=model, 
                test_loader=test_loader, 
                criterion=criterion, 
                logger=logger,
                device=device)
            )
        # valid_loss 최솟값 저장
        logger.info(f'TOTAL RUN {epoch}/{num_epochs - 1}, Train_Loss: {train_loss[-1]:.4f},  VAoid_Loss: {test_loss[-1]:.4f}')
        if test_loss[-1] < best_test_loss:
            best_test_loss = test_loss[-1]
            torch.save(model.state_dict(), path)
            logger.info(f'UPDATE best_test_loss :{best_test_loss:.4f}')
            
    return best_test_loss, train_loss, test_loss

def run_train_test_1_input(model, train_loader, test_loader, criterion, optimizer, num_epochs, test_step, logger, device, path):
    train_loss_arr = []
    test_loss_arr = []

    best_test_loss = 99999999
    early_stop, early_stop_max = 0., 3.

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.
        for batch_X, _ in train_loader:
        
            batch_X = batch_X.to(device)
            optimizer.zero_grad()

            # Forward Pass
            model.train()
            outputs = model(batch_X)
            train_loss = criterion(outputs, batch_X)
            epoch_loss += train_loss.data

            # Backward and optimize
            train_loss.backward()
            optimizer.step()

            train_loss_arr.append(epoch_loss / len(train_loader.dataset))

            if epoch % 10 == 0:
                model.eval()

            test_loss = 0.

        if epoch % test_step == 0:
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)

                # Forward Pass
                outputs = model(batch_X)
                batch_loss = criterion(outputs, batch_X)
                test_loss += batch_loss.data

            test_loss = test_loss
            test_loss_arr.append(test_loss)

            if best_test_loss > test_loss:
                best_test_loss = test_loss
                early_stop = 0

                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f} *'.format(epoch, num_epochs, epoch_loss, test_loss))
            else:
                early_stop += 1
                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, num_epochs, epoch_loss, test_loss))   

            if early_stop >= early_stop_max:
                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, num_epochs, epoch_loss, test_loss))   
                print('Early stopping!')
                retrn
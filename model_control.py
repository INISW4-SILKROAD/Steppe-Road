import torch 

def freeze_model(model, parent= '', exception = []):
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
        for image_em, portion, labels in train_loader:
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
def evaluate_model(model, val_loader, criterion, logger, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        batch = 0
        for image_em, portion, labels in val_loader:
            batch +=1
            logger.info(f'valid batch {batch} start')

            image_em, portion ,labels = image_em.to(device), portion.to(device), labels.to(device)
           
            outputs = model(image_em, portion)
            logger.info(f'train batch {batch} output : {outputs.shape} ')

            loss = criterion(outputs, labels)
            logger.info(f'train batch {batch} loss : {loss:.4f} ')
            
            val_loss += loss.item() * image_em.size(0)
            logger.info(f'valid batch {batch} end, running_loss: {val_loss:.4f}')
    
    val_loss /= len(val_loader.dataset)
    logger.info(f'Validation Loss: {val_loss:.4f}')
    
    return val_loss

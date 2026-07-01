import os
import torch
import numpy as np

from utils.logging import get_logger
from utils.exception import ValueError, TypeError

logger = get_logger(__name__)
logger.setLevel('WARNING')

class ModelTuner(object):
	def __init__(self, config=None):
		self._config=config
		self._model=None
		self._optimiser = None          # 主优化器（单阶段模式）
		self._vae_optimiser = None      # VAE 部分优化器（两阶段）
		self._bm_optimiser = None       # BM 部分优化器（两阶段）
		self._use_two_optimisers = False

		self.train_loader=None
		self.test_loader=None

		self.outpath=""
		self.infile=""

	def save_model(self,config_string='test'):
		logger.info("Saving Model")
		f=open(os.path.join(self.outpath,"model_{0}.pt".format(config_string)),'wb')
		torch.save(self._model.state_dict(),f)
		f.close()
		return
	
	def save_rbm(self,config_string='test'):
		logger.info("Saving RBM")
		f=open(os.path.join(self.outpath,"rbm_{0}.pt".format(config_string)),'wb')
		print(self._model.prior)
		torch.save(self._model.prior,f)
		f.close()
		return

	def register_model(self,model):
		logger.debug("Register Model")
		self._model=model
		return

	def register_optimiser(self,optimiser):
		"""单阶段优化器（兼容原有用法）"""
		logger.debug("Register Model")
		self._optimiser=optimiser
		self._use_two_optimisers = False
		return

	def register_two_optimisers(self, vae_optimiser, bm_optimiser):
		"""两阶段优化器（VAE 和 BM 分离）"""
		logger.debug("Register Model")
		self._vae_optimiser = vae_optimiser
		self._bm_optimiser = bm_optimiser
		self._use_two_optimisers = True
		return

	def register_dataLoaders(self,train_loader,test_loader):
		self.train_loader=train_loader
		self.test_loader=test_loader
		return
	
	def get_input_dimension(self):
		assert self.train_loader is not None, "Trying to retrieve datapoint from empty train loader"
		return self.train_loader.get_input_size()
	
	def get_train_dataset_mean(self):
		#returns mean of dataset as list
		#multiple input datasets - multiple means
		assert self.train_loader is not None, "Trying to retrieve datapoint from empty train loader"
		
		input_dimension=self.get_input_dimension()
		imgPerLayer={}	
		for i in range(0,len(input_dimension)):
			imgPerLayer[i]=[]	
		for i, (data, _) in enumerate(self.train_loader.dataset):
			#loop over all layers
			for l,d in enumerate(data):	
				imgPerLayer[l].append(d.view(-1,input_dimension[l]))
		means=[]
		for l, imgList in imgPerLayer.items():
			means.append(torch.mean(torch.stack(imgList),dim=0))
		return means

	def load_model(self,set_eval=True):
		logger.info("Loading Model")
		#attention: model must be defined already
		self._model.load_state_dict(torch.load(self.infile))
		#training of model
		if set_eval:
			self._model.eval()
		return

	def train(self, epoch):
		logger.info("Training Model")
		#set pytorch train mode
		self._model.train()

		total_train_loss = 0
		for batch_idx, (inputData, label) in enumerate(self.train_loader):
			#set gradients to zero before backprop. Needed in pytorch
			# self._optimiser.zero_grad()

			# 第一阶段：更新 VAE 参数
			if self._use_two_optimisers:
				self._vae_optimiser.zero_grad()
			else:
				self._optimiser.zero_grad()

			if self._config.type == 'QVAE':
				# 注意：forward 返回: output_logits, posterior_dist, q_logits, zeta
				output_logits, posterior, q, zeta = self._model(inputData)
				# 直接使用 forward 返回的 logits 和 posterior 计算损失
				train_loss = self._model.loss(inputData, output_logits, posterior)
			else:
				logger.debug("ERROR Unknown Model Type")
				raise NotImplementedError

			train_loss.backward()
			# total_train_loss += train_loss.item()
			# self._optimiser.step()
			if self._use_two_optimisers:
				self._vae_optimiser.step()
			else:
				self._optimiser.step()
			
			# 第二阶段：更新 BM 参数（仅在两阶段模式下）
			if self._use_two_optimisers and (self._config.type in ['QVAE', 'CellQVAE']):
				self._bm_optimiser.zero_grad()
				# 使用提取的 q 计算 BM 损失（注意 q 应已 detach，但为安全再次 detach）
				bm_loss = self._model.bm_loss(q.detach(), getattr(self._config, 'weight_decay', 0.0))
				bm_loss.backward()
				self._bm_optimiser.step()
				total_train_loss += train_loss.item() + bm_loss.item()  # 记录总损失
			else:
				total_train_loss += train_loss.item()

			# Output logging
			if batch_idx % 100 == 0:
				logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx*len(inputData), len(self.train_loader.dataset),
					100.*batch_idx/len(self.train_loader), train_loss.data.item()/len(inputData)))
		
		total_train_loss /= len(self.train_loader.dataset)
		logger.info("Train Loss: {0}".format(total_train_loss))
		return total_train_loss
	
	def test(self):
		logger.info("Testing Model")
		self._model.eval()

		test_loss = 0
		zeta_list=None
		label_list=None

		with torch.no_grad():
			for batch_idx, (inputData, label) in enumerate(self.test_loader):
				if self._config.type == 'QVAE':
					# forward 返回: output_logits, posterior, q, zeta
					output_logits, posterior, q, zeta = self._model(inputData)
					test_loss += self._model.loss(inputData, output_logits, posterior)

					# 设置 outputData 为概率值（用于绘图）
					outputData = torch.sigmoid(output_logits)
                    # # 记录 label_list
                    # label_list = label.detach().numpy() if label_list is None else np.append(label_list, label.detach().numpy(), axis=0)

		test_loss /= len(self.test_loader.dataset)
		logger.info("Test Loss: {0}".format(test_loss))
		return test_loss, inputData, outputData, label_list
import os
import csv
import Loss
from models import *
from torch import optim
from evaluation import *

class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.scheduler = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch

		# Losses
		self.criterion = Loss.DiceLoss()

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_test = config.num_epochs_test
		self.batch_size = config.batch_size

		# Path
		self.model_path = config.model_path
		self.train_result_path = config.train_result_path
		self.val_result_path = config.val_result_path
		self.test_result_path = config.test_result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=1,output_ch=1)
		elif self.model_type == 'MSU_Net':
			self.unet = MSU_Net(img_ch=1, output_ch=1)

		self.optimizer = optim.SGD(self.unet.parameters(), self.lr, momentum=0.9, weight_decay=0.000001)

		#self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

		self.unet.to(self.device)

		# self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def reset_grad(self):
		# Zero the gradient buffers.
		self.unet.zero_grad()

	def train(self):
		"""Train encoder, generator and discriminator."""

		# ====================================== Training ===========================================#

		unet_path = os.path.join(self.model_path, '%s-%d-%.4f.pkl' % (self.model_type, self.num_epochs, self.lr))

		if os.path.isfile(unet_path):
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
		else:
			learn_rate = self.lr
			best_Dice = 0.

			for epoch in range(self.num_epochs):

				self.unet.train()
				epoch_loss = 0

				t_TPR = 0.  # TPR
				t_FPR = 0.  # FPR
				t_PPV = 0.  # PPV
				t_JS = 0.   # Jaccard Similarity
				t_DC = 0.   # Dice Coefficient
				t_SE = 0.   # SE
				t_SP = 0.   # SP

				for i, (images, label) in enumerate(self.train_loader):

					images = images.to(self.device)
					GT = label.to(self.device)
					SR = F.sigmoid(self.unet(images))

					SR_flat = SR.view(SR.size(0), -1)
					GT_flat = GT.view(GT.size(0), -1)

					loss = self.criterion(SR_flat, GT_flat)

					epoch_loss += loss.item()

					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()

					t_TPR += get_TPR(SR, GT)
					t_FPR += get_FPR(SR, GT)
					t_PPV += get_precision(SR, GT)
					t_JS += get_JS(SR, GT)
					t_DC += get_DC(SR, GT)
					t_SE += get_sensitivity(SR, GT)
					t_SP += get_specificity(SR, GT)

				length = len(self.train_loader)
				t_TPR = t_TPR / length
				t_FPR = t_FPR / length
				t_PPV = t_PPV / length
				t_JS = t_JS / length
				t_DC = t_DC / length
				t_SE = t_SE / length
				t_SP = t_SP / length

				"""
				torchvision.utils.save_image(images.data.cpu(),
                                             os.path.join(self.train_result_path,
                                                          '%s_train_%d_image.jpg' % (self.model_type, epoch + 1)))
				torchvision.utils.save_image(SR.data.cpu(),
                                             os.path.join(self.train_result_path,
                                                          '%s_train_%d_SR.jpg' % (self.model_type, epoch + 1)))
				torchvision.utils.save_image(GT.data.cpu(),
                                             os.path.join(self.train_result_path,
                                                          '%s_train_%d_GT.jpg' % (self.model_type, epoch + 1)))
                """

	#===================================== Validation ====================================#
				self.unet.train(False)
				self.unet.eval()

				with torch.no_grad():

					v_TPR = 0.  # TPR
					v_FPR = 0.  # FPR
					v_PPV = 0.  # PPV
					v_JS = 0.  # Jaccard Similarity
					v_DC = 0.  # Dice Coefficient
					v_SE = 0.  # SE
					v_SP = 0.  # SP

					for i, (images, GT) in enumerate(self.valid_loader):

						images = images.to(self.device)
						GT = GT.to(self.device)
						SR = F.sigmoid(self.unet(images))

						v_TPR += get_TPR(SR, GT)
						v_FPR += get_FPR(SR, GT)
						v_PPV += get_precision(SR, GT)
						v_JS += get_JS(SR, GT)
						v_DC += get_DC(SR, GT)
						v_SE += get_sensitivity(SR, GT)
						v_SP += get_specificity(SR, GT)

					length = len(self.valid_loader)
					v_TPR = v_TPR / length
					v_FPR = v_FPR / length
					v_PPV = v_PPV / length
					v_JS = v_JS / length
					v_DC = v_DC / length
					v_SE = v_SE / length
					v_SP = v_SP / length

					"""
					self.scheduler.step()
					# learn_rate = self.scheduler.get_lr()
					zdy = self.scheduler.get_lr()
					learn_rate = zdy[0]

					"""
					# Print the log info
					print(
						'Epoch [%d/%d], \n[Training]   TPR: %.4f, FPR: %.4f, PPV:%.4f, JS: %.4f, DC: %.4f, SE:%.4f, SP:%.4f' % (
							epoch + 1, self.num_epochs, t_TPR, t_FPR, t_PPV, t_JS, t_DC, t_SE, t_SP))
					print(
						'[Validation] TPR: %.4f, FPR: %.4f, PPV:%.4f, JS: %.4f, DC: %.4f, SE:%.4f, SP:%.4f\n[Loss]: %.4f  [lr]: %.4f' % (
							v_TPR, v_FPR, v_PPV, v_JS, v_DC, v_SE, v_SP, epoch_loss, learn_rate))# , loss_sum_1, loss_sum_2, loss_sum_3

					"""
					torchvision.utils.save_image(images.data.cpu(),
                                                 os.path.join(self.val_result_path,
                                                              '%s_val_%d_image.jpg' % (self.model_type, epoch + 1)))
					torchvision.utils.save_image(SR.data.cpu(),
                                                 os.path.join(self.val_result_path,
                                                              '%s_val_%d_SR.jpg' % (self.model_type, epoch + 1)))
					torchvision.utils.save_image(GT.data.cpu(),
                                                 os.path.join(self.val_result_path,
                                                              '%s_val_%d_GT.jpg' % (self.model_type, epoch + 1)))
                    """

					e = open(os.path.join(self.train_result_path, 'train_result.csv'), 'a', encoding='utf-8',
							 newline='')
					wr = csv.writer(e)
					wr.writerow(
						[self.model_type, t_TPR, t_FPR, t_PPV, t_JS, t_DC,t_SE, t_SP, epoch_loss, epoch + 1, self.num_epochs,
						self.augmentation_prob, learn_rate])
					e.close()

					h = open(os.path.join(self.val_result_path, 'val_result.csv'), 'a', encoding='utf-8', newline='')
					wr = csv.writer(h)
					wr.writerow(
						[self.model_type, v_TPR, v_FPR, v_PPV, v_JS, v_DC, v_SE, v_SP, epoch + 1, self.num_epochs,
						self.augmentation_prob])
					h.close()

					# Save Best U-Net model
					if v_DC + v_JS > best_Dice:
						best_Dice = v_DC + v_JS
						best_unet = self.unet.state_dict()
						print('Best %s model score : %.4f' % (self.model_type, best_Dice))
						torch.save(best_unet, unet_path)


	# ===================================== Test ====================================#
	def test(self):

		unet_path = os.path.join(self.model_path, '%s-%d-%.4f.pkl' % (
			self.model_type, self.num_epochs, self.lr))

		self.unet.load_state_dict(torch.load(unet_path))
		print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))


		with torch.no_grad():
			for epoch in range(self.num_epochs_test):

				TPR = 0.  # TPR
				FPR = 0.  # FPR
				PPV = 0.   # PPV
				JS = 0.    # Jaccard Similarity
				DC = 0.    # Dice Coefficient
				SE = 0.
				SP = 0.

				for i, (images, GT) in enumerate(self.test_loader):

					images = images.to(self.device)

					GT = GT.to(self.device)
					SR = F.sigmoid(self.unet(images))

					TPR += get_TPR(SR, GT)
					FPR += get_FPR(SR, GT)
					PPV += get_precision(SR, GT)
					JS += get_JS(SR, GT)
					DC += get_DC(SR, GT)
					SE += get_sensitivity(SR, GT)
					SP += get_specificity(SR, GT)

				length = len(self.test_loader)
				TPR = TPR / length
				FPR = FPR / length
				PPV = PPV / length
				JS = JS/length
				DC = DC/length
				SE = SE / length
				SP = SP / length

				print('Epoch [%d/%d], \n[Test] TPR: %.4f, FPR: %.4f, PPV:%.4f, JS: %.4f, DC: %.4f, SE: %.4f, SP: %.4f'%
					  (epoch + 1, self.num_epochs_test, TPR, FPR, PPV, JS, DC,SE, SP))

				"""
				images = images * 0.5 + 0.5
				torchvision.utils.save_image(images.data.cpu(),
											 os.path.join(self.test_result_path,
													  '%s_test_%d_image.jpg' % (self.model_type, epoch + 1)))
				torchvision.utils.save_image(SR.data.cpu(),
											 os.path.join(self.test_result_path,
													  '%s_test_%d_SR.jpg' % (self.model_type, epoch + 1)))
				torchvision.utils.save_image(GT.data.cpu(),
											 os.path.join(self.test_result_path,
													  '%s_test_%d_GT.jpg' % (self.model_type, epoch + 1)))
				"""

				g = open(os.path.join(self.test_result_path,'test_result.csv'), 'a', encoding='utf-8', newline='')
				wr = csv.writer(g)
				wr.writerow([self.model_type,TPR,FPR, PPV, JS, DC, SE, SP, self.lr, epoch + 1, self.num_epochs_test ,self.augmentation_prob])
				g.close()
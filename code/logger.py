import logging
import torch
from tensorboardX import SummaryWriter
import os
import json
from tqdm import tqdm

import pdb


def get_logger(opts):
    return Logger(opts)


class Logger(object):
    def __init__(self, opts):
        self.opts = opts
        self.set_logger()

        self.write_opts_to_file()
        self.loss_train = torch.zeros((opts.n_epochs,))
        self.loss_test = torch.zeros((opts.n_epochs,))
        self.writer = SummaryWriter(logdir=opts.save_dir_tensorboard)

        self.logger.info('Experiment folder at %s' % opts.save_dir)

    def write_opts_to_file(self):
        fobj = open(os.path.join(self.opts.save_dir, 'options.txt'), 'w+')

        opts_dict = vars(self.opts)
        for k, v in opts_dict.items():
            fobj.write('{} >> {}\n'.format(str(k), str(v)))
        fobj.close()

    def log_model_graph(self, graph, input=None):
        self.writer.add_graph(graph, input_to_model=input, verbose=False)

    def log_ckpt(self, epoch, model):
        to_save = bool(self.loss_test[epoch] ==
                       torch.min(self.loss_test[:epoch+1])) if \
            not self.opts.save_every_ckpt else True

        if to_save:
            save_name = 'epoch-%d.model' % (epoch+1) \
                if self.opts.save_every_ckpt else 'epoch-best.model'
            torch.save(model.cpu().state_dict(), os.path.join(
                self.opts.save_dir_model, save_name))

            log_str = "Checkpoint saved at %s" % os.path.join(
                self.opts.save_dir_model, save_name
            )
            self.logger.info(log_str)

    def log_epoch(self, epoch, n_batches, train=True, model=None):
        if train:
            self.loss_train[epoch] /= float(n_batches)

            self.logger.info('Epoch [%3d/%3d]: Training Loss %.3f' % (
                epoch+1, self.opts.n_epochs, self.loss_train[epoch]
            ))
        else:
            self.loss_test[epoch] /= float(n_batches)
            self.writer.add_scalars('loss',
                                    {'val': self.loss_test[epoch]},
                                    (epoch + 1) * n_batches)

            self.logger.info('Epoch [%3d/%3d]: Testing Loss %.3f' % (
                epoch+1, self.opts.n_epochs, self.loss_test[epoch]
            ))

            self.log_ckpt(epoch, model)

    def log_iter(self, epoch, iter, loss, lr, train=True, model=None):
        if train:
            self.loss_train[epoch] += loss

            # write log to tensorboard and log.log in experiment folder
            if iter % self.opts.log_every == 0:
                self.writer.add_scalars('loss', {'training': loss}, iter)
                self.writer.add_scalar('learning_rate', lr, iter)

                log_str = ('Epoch [%3d/%3d], Iter [%3d/%3d]: loss = %.3f,'
                           ' lr = %.3f' % (
                               epoch, self.opts.n_epochs,
                               iter % self.opts.iter_per_epoch,
                               self.opts.iter_per_epoch, loss, lr
                           ))
                self.logger.debug(log_str)

            # optionally log gradients to tensorboard as well
            if (not (self.opts.hist_freq == -1)) and (model is not None):
                for name, param in model.named_parameters():
                    self.writer.add_histogram(
                        '%s_data' % name, param.cpu().data.numpy(), iter
                    )
                    self.writer.add_histogram(
                        '%s_grad' % name, param.cpu().grad.numpy(), iter
                    )
        else:
            self.loss_test[epoch] += loss

    def set_logger(self):

        class TqdmStream(object):
            @classmethod
            def write(_, msg):
                pass

        logging.basicConfig(stream=TqdmStream)
        self.logger = logging.getLogger(name='trainer')
        self.logger.setLevel(logging.DEBUG)

        # create handlers
        fh = logging.FileHandler(os.path.join(self.opts.save_dir, 'log.log'))
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '[%(asctime)s; %(levelname)s]: %(message)s'
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # add the handlers to logger
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    def terminate(self):
        pass

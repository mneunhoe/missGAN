require(zeallot)
# We need to pass our noise_dim and data_dim to create concrete networks
init_missGAN <- function(dat,
                         noise_dim = 2) {
  data_dim <- ncol(dat)

  # Now, we can set up a Generator net and send it to our device (cpu or gpu)
  g_net <-
    Generator(noise_dim, data_dim)$to(device = device)


  # To update the parameters of the network we need setup an optimizer. Here we use the adam optimizer with a learning rate of 0.0002
  g_optim <- torch::optim_adam(g_net$parameters, lr = 0.0002)

  # Now, we also need a Discriminator net.
  d_net <-
    Discriminator(data_dim = ncol(train_samples))$to(device = device)

  #To update the parameters of the network we need setup an optimizer. Here we use the adam optimizer with a learning rate of 0.0002 * 4
  # This heuristic comes from the idea of using two time-scales (aka different learning rates) for the Generator and Discriminator. You can find more in this paper: https://arxiv.org/abs/1706.08500
  d_optim <- torch::optim_adam(d_net$parameters, lr = 0.0002 * 4)

  # We need our real data in a torch tensor
  torch_data <-
    torch::torch_tensor(dat)$to(device = device)

  # To observe training we will also create one fixed noise data frame.
  # # torch_randn creates a torch object filled with draws from a standard normal distribution
  fixed_z <-
    torch::torch_randn(c(nrow(dat), noise_dim))$to(device = device)

  return(
    list(
      Generator = g_net,
      Discriminator = d_net,
      g_optimizer = g_optim,
      d_optimizer = d_optim,
      torch_data = torch_data,
      fixed_z = fixed_z,
      noise_dim = noise_dim

    )
  )

}

sample_synthetic_data <-
  function(g_net, z) {
    # Pass the noise through the Generator to create fake data
    fake_data <-  g_net(z)

    # Create an R array/matrix from the torch_tensor
    synth_data <- torch::as_array(fake_data$detach()$cpu())
    return(synth_data)
  }

GAN_training_loop <-
  function(GAN_nets,
           GAN_update_step,
           batch_size = 50,
           epochs = 10,
           monitor_training = FALSE) {
    # Steps: How many steps do we need to make before we see the entire data set (on average).
    steps <- nrow(GAN_nets$torch_data) %/% batch_size

    # Iters: What's the total number of update steps?
    iters <- steps * epochs

    for (step in 1:iters) {
      GAN_update_step(GAN_nets, batch_size)
      if(monitor_training) {
        if (step %% steps == 0) {
          # Print the current epoch to the console.
          cat("\n Done with Epoch: ", step %/% steps, "\n\n")

          # Create synthetic data for our plot. This synthetic data will always use the same noise sample -- fixed_z -- so it is easier for us to monitor training progress.
          synth_data <-
            sample_synthetic_data(GAN_nets$Generator, GAN_nets$fixed_z)
          # Now we plot the training data.
          plot(
            torch::as_array(GAN_nets$torch_data),
            bty = "n",
            col = viridis::viridis(2, alpha = 0.7)[1],
            pch = 19,
            xlab = "Var 1",
            ylab = "Var 2",
            main = paste0("Epoch: ", step %/% steps),
            las = 1
          )
          # And we add the synthetic data on top.
          points(
            synth_data,
            bty = "n",
            col = viridis::viridis(2, alpha = 0.7)[2],
            pch = 19
          )
          # Finally a legend to understand the plot.
          legend(
            "topleft",
            bty = "n",
            pch = 19,
            col = viridis::viridis(2),
            legend = c("Real", "Synthetic")
          )
        }
      }
    }
  }

GAN_update_step <-
  function(GAN_nets,
           batch_size = 50) {
    ##########################
    # Sample Batch of Data
    ###########################

    # For each training iteration we need a fresh (mini-)batch from our data.
    # So we first sample random IDs from our data set.
    batch_idx <-
      sample(nrow(GAN_nets$torch_data), size = batch_size)

    # Then we subset the data set (x is the torch version of the data) to our fresh batch.
    real_data <- GAN_nets$torch_data[batch_idx]$to(device = device)

    ###########################
    # Update the Discriminator
    ###########################

    # In a GAN we also need a noise sample for each training iteration.
    # torch_randn creates a torch object filled with draws from a standard normal distribution
    z <-
      torch::torch_randn(c(batch_size, GAN_nets$noise_dim))$to(device = device)

    # Now our Generator net produces fake data based on the noise sample.
    # Since we want to update the Discriminator, we do not need to calculate the gradients of the Generator net.
    fake_data <- torch::with_no_grad(GAN_nets$Generator(input = z))

    # The Discriminator net now computes the scores for fake and real data
    dis_real <- GAN_nets$Discriminator(real_data)
    dis_fake <- GAN_nets$Discriminator(fake_data)

    # We combine these scores to give our discriminator loss
    d_loss <- kl_real(dis_real) + kl_fake(dis_fake)
    d_loss <- d_loss$mean()

    # What follows is one update step for the Discriminator net

    # First set all previous gradients to zero
    GAN_nets$d_optimizer$zero_grad()

    # Pass the loss backward through the net
    d_loss$backward()

    # Take one step of the optimizer
    GAN_nets$d_optimizer$step()

    ###########################
    # Update the Generator
    ###########################

    # To update the Generator we will use a fresh noise sample.
    # torch_randn creates a torch object filled with draws from a standard normal distribution
    z <-
      torch::torch_randn(c(batch_size, GAN_nets$noise_dim))$to(device = device)

    # Now we can produce new fake data
    fake_data <- GAN_nets$Generator(z)

    # The Discriminator now scores the new fake data
    dis_fake <- GAN_nets$Discriminator(fake_data)

    # Now we can calculate the Generator loss
    g_loss = kl_gen(dis_fake)
    g_loss = g_loss$mean()

    # And take an update step of the Generator

    # First set all previous gradients to zero
    GAN_nets$g_optimizer$zero_grad()

    # Pass the loss backward through the net
    g_loss$backward()

    # Take one step of the optimizer
    GAN_nets$g_optimizer$step()

    cat("Discriminator loss: ",
        d_loss$item(),
        "\t Generator loss: ",
        g_loss$item(),
        "\n")
  }


init_weights <- function(m){
  if( "nn_linear" %in% attributes(m)$class){
    torch::nn_init_kaiming_normal_(m$weight$cpu())$to(device = device)
    m$bias$data()$cpu()$fill_(0)$to(device = device)
  }

}

init_missGAN2 <- function(dat, mask, transformer,
                         latent_dim = 2, IAF = T,
                         optimizer = "adam",
                         base_lr = 0.001, ttur_d_factor = 1,
                         n_encoder = list(256, 128),
                         n_decoder = list(128, 256),
                         ndf = list(256, 256),
                         encoder_dropout_rate = 0,
                         decoder_dropout_rate = 0,
                         D_dropout_rate = 0.5,
                         alpha = 10,
                         beta = 0.1,
                         gamma = 0.1,
                         pack = 1) {
  data_dim <- ncol(dat)

  # Now, we can set up a Generator net and send it to our device (cpu or gpu)
  if(IAF){
    encoder <-
    FlowEncoder(
      noise_dim = data_dim,
      data_dim = latent_dim,
      hidden_units = n_encoder,
      dropout_rate = encoder_dropout_rate
    )$to(device = device)
    } else {
  encoder <-
    Generator(
      noise_dim = data_dim,
      data_dim = latent_dim,
      hidden_units = n_encoder,
      dropout_rate = encoder_dropout_rate
    )$to(device = device)

  encoder$apply(init_weights)
}
  decoder <-
    Generator(
      noise_dim = latent_dim,
      data_dim = data_dim,
      hidden_units = n_decoder,
      dropout_rate = decoder_dropout_rate
    )$to(device = device)

  decoder$apply(init_weights)

  mask_decoder <-
    Generator(
      noise_dim = latent_dim,
      data_dim = data_dim,
      hidden_units = n_decoder,
      dropout_rate = decoder_dropout_rate
    )$to(device = device)

  mask_decoder$apply(init_weights)

  discriminator_d <-
    Discriminator(data_dim = data_dim,
                  hidden_units = ndf,
                  dropout_rate = D_dropout_rate,pack = pack)$to(device = device)

  discriminator_d$apply(init_weights)

  discriminator_e <-
    Discriminator(data_dim = latent_dim,
                  hidden_units = ndf,
                  dropout_rate = D_dropout_rate, pack = pack)$to(device = device)

  discriminator_e$apply(init_weights)

  # To update the parameters of the network we need setup an optimizer. Here we use the adam optimizer with a learning rate of 0.0002
  if(optimizer == "adam") {
    d_optim <- torch::optim_adam(do.call(c, list(discriminator_d$parameters,
                                                     discriminator_e$parameters)),
                                     lr = base_lr * ttur_d_factor, betas = c(0.5, 0.9))
    g_optim <- torch::optim_adam(do.call(c, list(encoder$parameters,
                                                     decoder$parameters,
                                                     mask_decoder$parameters)),
                                     lr = base_lr, betas = c(0.5, 0.9))
  }
  if(optimizer == "adamw") {
    d_optim <- torch::optim_adam(do.call(c, list(discriminator_d$parameters,
                                                 discriminator_e$parameters)),
                                 lr = base_lr * ttur_d_factor, weight_decay = 0.1, amsgrad = T)
    g_optim <- torch::optim_adam(do.call(c, list(encoder$parameters,
                                                 decoder$parameters,
                                                 mask_decoder$parameters)),
                                 lr = base_lr, weight_decay = 0.1, amsgrad = T)
  }

  criterionCycle <- torch::nn_mse_loss()
  MSEloss <- torch::nn_mse_loss()
  criterionCE <- torch::nn_cross_entropy_loss()
  criterionBCE <- torch::nn_bce_loss()



  # To observe training we will also create one fixed noise data frame.
  # # torch_randn creates a torch object filled with draws from a standard normal distribution
  fixed_z <-
    torch::torch_randn(c(nrow(dat), latent_dim))$to(device = device)

  return(
    list(
      encoder = encoder,
      decoder = decoder,
      mask_decoder = mask_decoder,
      discriminator_d = discriminator_d,
      discriminator_e = discriminator_e,
      g_optim = g_optim,
      d_optim = d_optim,
      torch_data = dat,
      torch_mask = mask,
      transformer = transformer,
      fixed_z = fixed_z,
      latent_dim = latent_dim,
      losses = list(criterionCycle, MSEloss, criterionCE, criterionBCE),
      g_loss_weights = list(alpha, beta, gamma)

    )
  )

}



GAN2_update_step <-
  function(GAN_nets,
           loss = "wgan_gp",
           batch_size = 50) {
    ##########################
    # Sample Batch of Data
    ###########################

    # For each training iteration we need a fresh (mini-)batch from our data.
    # So we first sample random IDs from our data set.
    batch_idx <-
      sample(nrow(GAN_nets$torch_data), size = batch_size)

    # Then we subset the data set (x is the torch version of the data) to our fresh batch.
    real_data <- GAN_nets$torch_data[batch_idx]$to(device = device)
    real_mask <- GAN_nets$torch_mask[batch_idx]$to(device = device)

    ###########################
    # Update the Discriminator
    ###########################

    for(p in GAN_nets$encoder$parameters){
      p$requires_grad_(F)
    }
    for(p in GAN_nets$decoder$parameters){
      p$requires_grad_(F)
    }
    for(p in GAN_nets$mask_decoder$parameters){
      p$requires_grad_(F)
    }

    noise <- torch::torch_randn(real_mask$shape)$to(device = device)
    #noise <- torch::torch_zeros(real_mask$shape)$to(device = device)
    noise$requires_grad <- T
    z_enc <- GAN_nets$encoder(real_mask*real_data + (1-real_mask)*noise)
    #z_enc <- GAN_nets$encoder(real_data*real_mask)
    z_gen <- torch::torch_empty_like(z_enc)$cpu()$normal_()$to(device = device)

    x_gen <- apply_activate(GAN_nets$decoder(z_gen), GAN_nets$transformer)
    x_rec <- apply_activate(GAN_nets$decoder(z_enc), GAN_nets$transformer)

    fake_mask <- apply_mask_activate(GAN_nets$mask_decoder(z_gen))
    mask_rec <- apply_mask_activate(GAN_nets$mask_decoder(z_enc))
    
    if(loss == "wgan_gp"){
      
      real_d_score <- GAN_nets$discriminator_d(real_mask*real_data + (1-real_mask)*noise)
      fake_d_score <- GAN_nets$discriminator_d(fake_mask*x_gen + (1-fake_mask)*noise)
    
      pen_d <- GAN_nets$discriminator_d$calc_gradient_penalty(GAN_nets$discriminator_d,
                        real_mask*real_data + (1-real_mask)*noise, fake_mask*x_gen + (1-fake_mask)*noise, device = device)
      loss_d <- -(torch::torch_mean(real_d_score) - torch::torch_mean(fake_d_score))
    
      fake_e_score <- GAN_nets$discriminator_e(z_enc)
      real_e_score <- GAN_nets$discriminator_e(z_gen)
      
      pen_e <- GAN_nets$discriminator_e$calc_gradient_penalty(GAN_nets$discriminator_e,
                        z_gen, z_enc, device = device)
      loss_e <- -(torch::torch_mean(real_e_score) - torch::torch_mean(fake_e_score))
      
      
      
    D_loss <-  loss_d*0.5 + loss_e*0.5 + pen_d + pen_e
    D_loss <- D_loss$mean()
    D_loss$requires_grad <- T

    GAN_nets$d_optim$zero_grad()
    D_loss$backward()
    GAN_nets$d_optim$step()
      
      
    } else {
    real_d_score <- GAN_nets$discriminator_d(real_mask*real_data + (1-real_mask)*noise)
    fake_d_score <- GAN_nets$discriminator_d(fake_mask*x_gen + (1-fake_mask)*noise)

    fake_e_score <- GAN_nets$discriminator_e(z_enc)
    real_e_score <- GAN_nets$discriminator_e(z_gen)

    loss_d_real <- kl_real(real_d_score)
    loss_d_fake <- kl_fake(fake_d_score)

    loss_e_real <- kl_real(real_e_score)
    loss_e_fake <- kl_fake(fake_e_score)
    
    loss_d <- (loss_d_real + loss_d_fake)*0.5
    loss_e <- (loss_e_real + loss_e_fake)*0.5
    D_loss <-  loss_d + loss_e
    D_loss <- D_loss$mean()
    D_loss$requires_grad <- T

    GAN_nets$d_optim$zero_grad()
    D_loss$backward()
    GAN_nets$d_optim$step()
}
    ###########################
    # Update the Generator
    ###########################
    for(p in GAN_nets$encoder$parameters){
      p$requires_grad_(T)
    }
    for(p in GAN_nets$decoder$parameters){
      p$requires_grad_(T)
    }
    for(p in GAN_nets$mask_decoder$parameters){
      p$requires_grad_(T)
    }
    for(p in GAN_nets$discriminator_d$parameters){
      p$requires_grad_(F)
    }
    for(p in GAN_nets$discriminator_e$parameters){
      p$requires_grad_(F)
    }

    # To update the Generator we will use a fresh noise sample.
    # torch_randn creates a torch object filled with draws from a standard normal distribution
    noise <- torch::torch_randn(real_mask$shape)$to(device = device)
    z_enc <- GAN_nets$encoder(real_mask*real_data + (1-real_mask)*noise)
    #z_enc <- GAN_nets$encoder(real_mask*real_data)
    z_gen <- torch::torch_empty_like(z_enc)$cpu()$normal_()$to(device = device)

    x_gen <- apply_activate(GAN_nets$decoder(z_gen), GAN_nets$transformer)
    x_rec <- apply_activate(GAN_nets$decoder(z_enc), GAN_nets$transformer)

    fake_mask <- apply_mask_activate(GAN_nets$mask_decoder(z_gen))
    mask_rec <- apply_mask_activate(GAN_nets$mask_decoder(z_enc))

    z_rec <- GAN_nets$encoder(fake_mask*x_gen + (1-fake_mask)*noise)

    start_idx <- 1
    ae_loss <- torch::torch_zeros(1, requires_grad = T)$to(device = device)

    for(info in GAN_nets$transformer$output_info){
      i <- info[[1]]
      if(info[[2]] == "linear" | info[[2]] == "tanh") {
        ae_loss <- ae_loss + GAN_nets$losses[[1]](x_rec[,start_idx:(start_idx+i-1)][real_mask[,start_idx:(start_idx+i-1)]==1],
                                                  real_data[,start_idx:(start_idx+i-1)][real_mask[,start_idx:(start_idx+i-1)]==1])
      } else if(info[[2]] == "softmax") {
        ae_loss <- ae_loss + GAN_nets$losses[[3]](x_rec[,start_idx:(start_idx+i-1)][real_mask[,start_idx:(start_idx+i-1)]==1]$view(c(-1, i)),
                                                  torch::torch_argmax(real_data[,start_idx:(start_idx+i-1)][real_mask[,start_idx:(start_idx+i-1)]==1]$view(c(-1, i)), dim = 2))
      } else {
        ae_loss <- ae_loss + GAN_nets$losses[[4]](x_rec[,start_idx:(start_idx+i-1)],
                                                  real_data[,start_idx:(start_idx+i-1)])
      }
      start_idx <- start_idx + i
    }

    z_ae_loss <- GAN_nets$losses[[2]](z_rec, z_gen)
    mask_ae_loss <- GAN_nets$losses[[4]](mask_rec, real_mask)

    fake_d_score <- GAN_nets$discriminator_d(fake_mask*x_gen + (1-fake_mask)*noise)
    fake_e_score <- GAN_nets$discriminator_e(z_enc)
    if(loss == "wgan_gp"){
      G_loss_d <- -torch::torch_mean(fake_d_score)
      G_loss_e <- -torch::torch_mean(fake_e_score)
   } else {
    G_loss_d <- kl_gen(fake_d_score)
    G_loss_e <- kl_gen(fake_e_score)
}
    G_loss1 <- G_loss_d + G_loss_e
    G_loss <- G_loss1 + GAN_nets$g_loss_weights[[1]] * ae_loss + GAN_nets$g_loss_weights[[2]] * z_ae_loss + GAN_nets$g_loss_weights[[3]] * mask_ae_loss

    GAN_nets$g_optim$zero_grad()
    G_loss$backward()
    GAN_nets$g_optim$step()

    cat("Discriminator loss: ",
        D_loss$item(),
        "\t Generator loss: ",
        G_loss$item(),
        "\n")
    
    return(list(D_loss$item(), G_loss$item()))
  }


init_missGAN3 <- function(dat, mask, transformer,
                         latent_dim = 2, IAF = T,
                         optimizer = "adam",
                         base_lr = 0.001, ttur_d_factor = 1,
                         n_encoder = list(128, 128, 128),
                         n_decoder = list(128, 128, 128),
                         ndf = list(128, 128, 128),
                         encoder_dropout_rate = 0,
                         decoder_dropout_rate = 0,
                         D_dropout_rate = 0,
                         alpha = 10,
                         beta = 0.1,
                         gamma = 0.1,
                         pack = 1) {
  data_dim <- ncol(dat)

  # Now, we can set up a Generator net and send it to our device (cpu or gpu)
  encoder <-
    Encoder(
      noise_dim = data_dim,
      data_dim = latent_dim,
      hidden_units = n_encoder,
      dropout_rate = encoder_dropout_rate
    )$to(device = device)

  encoder$apply(init_weights)

  decoder <-
    Generator(
      noise_dim = latent_dim,
      data_dim = data_dim,
      hidden_units = n_decoder,
      dropout_rate = decoder_dropout_rate
    )$to(device = device)

  decoder$apply(init_weights)

  mask_decoder <-
    Generator(
      noise_dim = latent_dim,
      data_dim = data_dim,
      hidden_units = n_decoder,
      dropout_rate = decoder_dropout_rate
    )$to(device = device)

  mask_decoder$apply(init_weights)

  discriminator_d <-
    Discriminator(data_dim = data_dim,
                  hidden_units = ndf,
                  dropout_rate = D_dropout_rate,pack = pack, sigmoid = T)$to(device = device)

  discriminator_d$apply(init_weights)

  discriminator_e <-
    Discriminator(data_dim = latent_dim,
                  hidden_units = ndf,
                  dropout_rate = D_dropout_rate, pack = pack, sigmoid = T)$to(device = device)

  discriminator_e$apply(init_weights)

  # To update the parameters of the network we need setup an optimizer. Here we use the adam optimizer with a learning rate of 0.0002
  if(optimizer == "adam") {
    d_optim <- torch::optim_adam(do.call(c, list(discriminator_d$parameters,
                                                     discriminator_e$parameters)),
                                     lr = base_lr * ttur_d_factor, betas = c(0.5, 0.9))
    g_optim <- torch::optim_adam(do.call(c, list(encoder$parameters,
                                                     decoder$parameters,
                                                     mask_decoder$parameters)),
                                     lr = base_lr, betas = c(0.5, 0.9))
  }
  if(optimizer == "adamw") {
    d_optim <- torch::optim_adam(do.call(c, list(discriminator_d$parameters,
                                                 discriminator_e$parameters)),
                                 lr = base_lr * ttur_d_factor, weight_decay = 0.1, amsgrad = T)
    g_optim <- torch::optim_adam(do.call(c, list(encoder$parameters,
                                                 decoder$parameters,
                                                 mask_decoder$parameters)),
                                 lr = base_lr, weight_decay = 0.1, amsgrad = T)
  }

  criterionCycle <- torch::nn_mse_loss()
  MSEloss <- torch::nn_mse_loss()
  criterionCE <- torch::nn_cross_entropy_loss()
  criterionBCE <- torch::nn_bce_loss()



  # To observe training we will also create one fixed noise data frame.
  # # torch_randn creates a torch object filled with draws from a standard normal distribution
  fixed_z <-
    torch::torch_randn(c(nrow(dat), latent_dim))$to(device = device)

  return(
    list(
      encoder = encoder,
      decoder = decoder,
      mask_decoder = mask_decoder,
      discriminator_d = discriminator_d,
      discriminator_e = discriminator_e,
      g_optim = g_optim,
      d_optim = d_optim,
      torch_data = dat,
      torch_mask = mask,
      transformer = transformer,
      fixed_z = fixed_z,
      latent_dim = latent_dim,
      losses = list(criterionCycle, MSEloss, criterionCE, criterionBCE),
      g_loss_weights = list(alpha, beta, gamma)

    )
  )

}

gauss_repara <- function(mu, logvar, n_sample = 1) {
  std <- logvar$mul(0.5)$exp_()
  size <- std$size()

  eps <- torch::torch_randn(shape = c(size[1], n_sample, size[2]), requires_grad = T, device = device)

  z <- eps$mul(std$reshape(c(size[1], n_sample, size[2])))$add_(mu$reshape(c(size[1], n_sample, size[2])))

  z <- torch::torch_clamp(z, -6, 6)

  return(z$view(list(z$size(1)*z$size(2), z$size(3))))

}

log_prob_gaussian <- function(z, mu, log_var){
  res <- - 0.5 * log_var - ((z - mu)^2.0 / (2.0 * torch::torch_exp(log_var)))
res <- res - 0.5 * log(2*pi)
return(res)
}



kld_std_guss <- function(mu, log_var){

kld = -0.5 * torch::torch_sum(log_var + 1. - mu^2 - torch::torch_exp(log_var), dim=2)
return(kld)
}



GAN3_update_step <-
  function(GAN_nets,
           loss = "gan",
           batch_size = 50) {
    ##########################
    # Sample Batch of Data
    ###########################

    # For each training iteration we need a fresh (mini-)batch from our data.
    # So we first sample random IDs from our data set.
    batch_idx <-
      sample(nrow(GAN_nets$torch_data), size = batch_size)

    # Then we subset the data set (x is the torch version of the data) to our fresh batch.
    real_data <- GAN_nets$torch_data[batch_idx]$to(device = device)
    real_mask <- GAN_nets$torch_mask[batch_idx]$to(device = device)

    ###########################
    # Update the Discriminator
    ###########################

    for(p in GAN_nets$encoder$parameters){
      p$requires_grad_(F)
    }
    for(p in GAN_nets$decoder$parameters){
      p$requires_grad_(F)
    }
    for(p in GAN_nets$mask_decoder$parameters){
      p$requires_grad_(F)
    }

    #noise <- torch::torch_randn(real_mask$shape)$to(device = device)
    #noise <- torch::torch_zeros(real_mask$shape)$to(device = device)
    #noise$requires_grad <- T
    #z_enc <- GAN_nets$encoder(real_mask*real_data + (1-real_mask)*noise)
    c(mu_z_enc, var_z_enc) %<-% GAN_nets$encoder(real_data*real_mask)
    z_enc <- gauss_repara(mu_z_enc, var_z_enc)
    #z_gen <- torch::torch_empty_like(z_enc)$cpu()$normal_()$to(device = device)
    z_gen <- torch::torch_randn_like(z_enc, device = device)
    
    x_gen <- apply_activate(GAN_nets$decoder(z_gen), GAN_nets$transformer)
    #x_rec <- apply_activate(GAN_nets$decoder(z_enc), GAN_nets$transformer)

    fake_mask <- apply_mask_activate(GAN_nets$mask_decoder(z_gen), temperature = 1)

    eps_fake <- torch::torch_rand_like(fake_mask, device = device)
    fake_mask <- torch::nnf_sigmoid((torch::torch_log(fake_mask) + torch::torch_log(eps_fake) - torch::torch_log(1 - eps_fake))/0.1)
    

    #mask_rec <- apply_mask_activate(GAN_nets$mask_decoder(z_enc), temperature = 1)

    #eps_rec <- torch::torch_rand_like(mask_rec, device = device)
    #mask_rec <- nnf_sigmoid((torch_log(mask_rec) + torch_log(eps_rec) - torch_log(1 - eps_rec))/0.1)

    
    if(loss == "wgan_gp"){
      
      real_d_score <- GAN_nets$discriminator_d(real_mask*real_data)
      fake_d_score <- GAN_nets$discriminator_d(fake_mask*x_gen )
    
      pen_d <- GAN_nets$discriminator_d$calc_gradient_penalty(GAN_nets$discriminator_d,
                        real_mask*real_data, fake_mask*x_gen, device = device)
      loss_d <- -(torch::torch_mean(real_d_score) - torch::torch_mean(fake_d_score))
    
      fake_e_score <- GAN_nets$discriminator_e(z_enc)
      real_e_score <- GAN_nets$discriminator_e(z_gen)
      
      pen_e <- GAN_nets$discriminator_e$calc_gradient_penalty(GAN_nets$discriminator_e,
                        z_gen, z_enc, device = device)
      loss_e <- -(torch::torch_mean(real_e_score) - torch::torch_mean(fake_e_score))
      
      
      
    D_loss <-  loss_d*0.5 + loss_e*0.5 + pen_d + pen_e
    D_loss <- D_loss$mean()
    D_loss$requires_grad <- T

    GAN_nets$d_optim$zero_grad()
    D_loss$backward()
    GAN_nets$d_optim$step()
      
      
    } else if(loss == "gan"){

      real_d_score <- GAN_nets$discriminator_d(real_mask*real_data)
      fake_d_score <- GAN_nets$discriminator_d(fake_mask*x_gen)
      
      loss_d <- 0.5 * (
      torch::nnf_binary_cross_entropy(fake_d_score, torch::torch_zeros_like(fake_d_score, device = device)) +
        torch::nnf_binary_cross_entropy(real_d_score, torch::torch_ones_like(real_d_score, device = device))
    )

      fake_e_score <- GAN_nets$discriminator_e(z_enc)
      real_e_score <- GAN_nets$discriminator_e(z_gen)
      
      loss_e <- 0.5 * (
      torch::nnf_binary_cross_entropy(fake_e_score, torch::torch_zeros_like(fake_e_score, device = device)) +
        torch::nnf_binary_cross_entropy(real_e_score, torch::torch_ones_like(real_e_score, device = device))
    )
      
      
      
    D_loss <-  loss_d + loss_e
    D_loss <- D_loss$mean()
    D_loss$requires_grad <- T

    GAN_nets$d_optim$zero_grad()
    D_loss$backward()
    GAN_nets$d_optim$step()

      } else{
    real_d_score <- GAN_nets$discriminator_d(real_mask*real_data)
    fake_d_score <- GAN_nets$discriminator_d(fake_mask*x_gen)

    fake_e_score <- GAN_nets$discriminator_e(z_enc)
    real_e_score <- GAN_nets$discriminator_e(z_gen)

    loss_d_real <- kl_real(real_d_score)
    loss_d_fake <- kl_fake(fake_d_score)

    loss_e_real <- kl_real(real_e_score)
    loss_e_fake <- kl_fake(fake_e_score)
    
    loss_d <- (loss_d_real + loss_d_fake)*0.5
    loss_e <- (loss_e_real + loss_e_fake)*0.5
    D_loss <-  loss_d + loss_e
    D_loss <- D_loss$mean()
    D_loss$requires_grad <- T

    GAN_nets$d_optim$zero_grad()
    D_loss$backward()
    GAN_nets$d_optim$step()
}
    ###########################
    # Update the Generator
    ###########################
    for(p in GAN_nets$encoder$parameters){
      p$requires_grad_(T)
    }
    for(p in GAN_nets$decoder$parameters){
      p$requires_grad_(T)
    }
    for(p in GAN_nets$mask_decoder$parameters){
      p$requires_grad_(T)
    }
    for(p in GAN_nets$discriminator_d$parameters){
      p$requires_grad_(F)
    }
    for(p in GAN_nets$discriminator_e$parameters){
      p$requires_grad_(F)
    }

    c(mu_z_enc, var_z_enc) %<-% GAN_nets$encoder(real_data*real_mask)
    z_enc <- gauss_repara(mu_z_enc, var_z_enc)
    #z_gen <- torch::torch_empty_like(z_enc)$cpu()$normal_()$to(device = device)
    z_gen <- torch::torch_randn_like(z_enc, device = device)
    
    x_gen <- apply_activate(GAN_nets$decoder(z_gen), GAN_nets$transformer)
    x_rec <- apply_activate(GAN_nets$decoder(z_enc), GAN_nets$transformer)

    fake_mask <- apply_mask_activate(GAN_nets$mask_decoder(z_gen), temperature = 1)

    eps_fake <- torch::torch_rand_like(fake_mask, device = device)
    fake_mask <- torch::nnf_sigmoid((torch::torch_log(fake_mask) + torch::torch_log(eps_fake) - torch::torch_log(1 - eps_fake))/0.1)
    

    mask_rec <- apply_mask_activate(GAN_nets$mask_decoder(z_enc), temperature = 1)

    eps_rec <- torch::torch_rand_like(mask_rec, device = device)
    mask_rec <- torch::nnf_sigmoid((torch::torch_log(mask_rec) + torch::torch_log(eps_rec) - torch::torch_log(1 - eps_rec))/0.1)

    c(mu_z_rec, var_z_rec) %<-% GAN_nets$encoder(fake_mask*x_gen)
    z_rec <- gauss_repara(mu_z_rec, var_z_rec)

    start_idx <- 1
    ae_loss <- torch::torch_zeros(1, requires_grad = T)$to(device = device)

    for(info in GAN_nets$transformer$output_info){
      i <- info[[1]]
      if(info[[2]] == "linear" | info[[2]] == "tanh") {
        ae_loss <- ae_loss + GAN_nets$losses[[1]](x_rec[,start_idx:(start_idx+i-1)][real_mask[,start_idx:(start_idx+i-1)]==1],
                                                  real_data[,start_idx:(start_idx+i-1)][real_mask[,start_idx:(start_idx+i-1)]==1])
      } else if(info[[2]] == "softmax") {
        ae_loss <- ae_loss + GAN_nets$losses[[3]](x_rec[,start_idx:(start_idx+i-1)][real_mask[,start_idx:(start_idx+i-1)]==1]$view(c(-1, i)),
                                                  torch::torch_argmax(real_data[,start_idx:(start_idx+i-1)][real_mask[,start_idx:(start_idx+i-1)]==1]$view(c(-1, i)), dim = 2))
      } else {
        ae_loss <- ae_loss + GAN_nets$losses[[4]](x_rec[,start_idx:(start_idx+i-1)],
                                                  real_data[,start_idx:(start_idx+i-1)])
      }
      start_idx <- start_idx + i
    }

    #z_ae_loss <- GAN_nets$losses[[2]](z_rec, z_gen)

  log_prob_z <- log_prob_gaussian(z_gen,
                                  mu_z_rec, 
                                  var_z_rec)
  
  z_ae_loss <- -1.0 * log_prob_z$mean(2)$mean(1)

  mask_ae_loss <- GAN_nets$losses[[4]](mask_rec, real_mask)

    fake_d_score <- GAN_nets$discriminator_d(fake_mask*x_gen)
    fake_e_score <- GAN_nets$discriminator_e(z_enc)
    if(loss == "wgan_gp"){
      G_loss_d <- -torch::torch_mean(fake_d_score)
      G_loss_e <- -torch::torch_mean(fake_e_score)
   } else if(loss == "gan"){
    G_loss_d <- torch::nnf_binary_cross_entropy(fake_d_score, torch::torch_ones_like(fake_d_score, device = device))
      G_loss_e <- torch::nnf_binary_cross_entropy(fake_e_score, torch::torch_ones_like(fake_e_score, device = device))
    } else{
    G_loss_d <- kl_gen(fake_d_score)
    G_loss_e <- kl_gen(fake_e_score)
}
    G_loss1 <- G_loss_d + G_loss_e
    G_loss <- G_loss1 + GAN_nets$g_loss_weights[[1]] * ae_loss + GAN_nets$g_loss_weights[[2]] * z_ae_loss + GAN_nets$g_loss_weights[[3]] * mask_ae_loss

    GAN_nets$g_optim$zero_grad()
    G_loss$backward()
    GAN_nets$g_optim$step()

    cat("Discriminator loss: ",
        D_loss$item(),
        "\t Generator loss: ",
        G_loss$item(),
        "\n")
    
    return(list(D_loss$item(), G_loss$item()))
  }



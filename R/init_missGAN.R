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

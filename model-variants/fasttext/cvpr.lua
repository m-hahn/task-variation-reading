require 'nn';



-- next steps

--- recurrent, two networks

--- LSTM



dataset={};
function dataset:size() return 100 end -- 100 examples
for i=1,dataset:size() do 
  local input = torch.randn(2);     -- normally distributed example in 2d
  local output = torch.Tensor(1);
  if input[1]*input[2]>0 then     -- calculate label for XOR function
    output[1] = -1;
  else
    output[1] = 1
  end
  dataset[i] = {input, output+1}
end



net = nn.Sequential()
net:add(nn.Linear(2, 5))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.Tanh())
net:add(nn.Linear(5, 7))
net:add(nn.Tanh())
net:add(nn.Linear(7, 1))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.Tanh())
--net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems

print('Lenet5\n' .. net:__tostring());


criterion = nn.MSECriterion() 
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)

print("END")





-------

trainset =  {{torch.Tensor(10),1},{torch.Tensor(10),2},{torch.Tensor(10),1}}
testset =  {{torch.Tensor(10),1}}
-- ignore setmetatable for now, it is a feature beyond the scope of this tutorial. It sets the index operator.
--setmetatable(trainset, 
--    {__index = function(t, i) 
--                    return t[i]
--                end}
--);
--trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self:size(1) 
end


input = torch.rand(10) -- pass a random tensor as input to the network
output = net:forward(input)
print(output)
net:zeroGradParameters() -- zero the internal gradient buffers of the network (will come to this later)
gradInput = net:backward(input, torch.rand(10))
print(#gradInput)



criterion = nn.ClassNLLCriterion() -- a negative log-likelihood criterion for multi-class classification
criterion:forward(output, 3) -- let's say the groundtruth was class number: 3
gradients = criterion:backward(output, 3)

gradInput = net:backward(input, gradients)

print(53)

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 1 -- just do 5 epochs of training.

print(59)

trainer:train(dataset)

print(63)





require 'nn'

local QLayer, Parent = torch.class('nn.QLayer', 'nn.Module')

function QLayer:__init(choices)
   -- the number of actions available in each time step
   self.choices = choices
   Parent.__init(self)
   self.output = torch.Tensor()
   self.resval = torch.Tensor()
end



-- input: tensor batch_size * N of Q values for choices
-- output: tensor batch_size of choices
function QLayer:updateOutput(input)
   --print(input[1])
   
   self.output:resize(input:size(1))
   self.resval:resize(input:size(1))
-- transform input such that:
-- (1) fill with arg max over Q value
-- (2) take a bit off for epsilon-greedy

   torch.max(self.resval, self.output, input, 2)


   epsilon = 0.2

   self.output:apply(function(choice)
         -- the first part is hard-coded here... this actually should be a policy given as a parameter
         if torch.uniform() < epsilon then
            return math.min(self.choices, torch.floor((torch.uniform() * self.choices)) + 1)
         elseif FIXED_ATTENTION > 0 and torch.uniform() < FIXED_ATTENTION then
            return 2
         elseif FIXED_ATTENTION < 0 and torch.uniform() < -FIXED_ATTENTION then
            return 1
         else
           return choice
         end
      end)
--[[   if torch.uniform() > 0.999 then
     print(input)
     print(self.output)
   end]]

  return self.output
end

-- the gradient of the probability (the second output) with regard to the attention values

function QLayer:updateGradInput(input, gradOutput)
   return nil
end



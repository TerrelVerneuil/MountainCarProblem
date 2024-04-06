import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

public class MountainCarProblem {
   //episodic semi-gradient control

   // Mountain Car, a standard testing domain in Reinforcement learning,
   // is a problem in which an under-powered car must drive up a steep
   // hill. Since gravity is stronger than the car's engine, even at full
   // throttle, the car cannot simply accelerate up the steep slope. The
   // car is situated in a valley and must learn to leverage potential
   // energy by driving up the opposite hill before the car is able to
   // make it to the goal at the top of the rightmost hill. The domain
   // has been used as a test bed in various reinforcement learning papers.

   // Variations
   // There are many versions of the mountain car
   // which deviate in different ways from the standard
   // model. Variables that vary include but are not
   // limited to changing the constants (gravity and steepness)
   // of the problem so specific tuning for specific policies
   // become irrelevant and altering the reward function to
   // affect the agent's ability to learn in a different manner.
   // An example is changing the reward to be equal to the
   // distance from the goal, or changing the reward to zero
   // everywhere and one at the goal. Additionally, a 3D
   // mountain car can be used, with a 4D continuous state space.

   // position and velocity
   double position = -0.5; //inital state //reset for every iteration
   double velocity = 0.0; // starting condition
   int numIter;
   int successes = 0;
   int totalAttempts = 0;
   Random random = new Random();
   
   double minPosition = -1.2;
   double maxPosition = 0.6;
   double minVelocity = -0.07;
   double maxVelocity = 0.07;
   int numStates = 50; //lower number of states more efficent path to goals but inconsistent learning
   int[] Action = { -1, 0, 1 };
   int reward = -1;
   Map<Integer, double[]> q = new TreeMap<>();
   // position = -0.5
   // velocity = 0.0
   // termination condition = (position >= 0.6)
   // drive left or right or not using engine
   List<Integer> stepsToGoal = new ArrayList<>();
   class State implements Cloneable {
       public State(int left, int neutral, int right) {
           
       }

       public double[] getQS(State s) {
           int hashCode = s.hashCode();
           double[] qActions = q.get(hashCode);
           if(qActions == null) {
               qActions = new double[3];
               q.put(hashCode, qActions);
           }
           return qActions;
       }


       public boolean isTerminal(double position, double maxPosition){
           return position >= maxPosition;
       }
   }
   public int getEpsilonGreedy(int state, double epsilon) {
       if (random.nextDouble() < epsilon) {
           return random.nextInt(Action.length); //random arbritary number 
       } else {
           double[] actions = q.get(state); //continue to figure out the best action
           int bestActionIndex = 0;
           double maxQValue = actions[0];
           for (int i = 1; i < actions.length; i++) {
               if (actions[i] > maxQValue) {
                   maxQValue = actions[i];
                   bestActionIndex = i;
               }
           }
           return bestActionIndex;
       }
   }
   public int discretizedPosition(double position) {
       return Math.min(numStates - 1, Math.max(0, (int) Math.floor((position - minPosition) / (maxPosition - minPosition) * numStates)));
   }
   public int discretizedVelocity(double velocity) {
       return Math.min(numStates - 1, Math.max(0, (int) Math.floor((velocity - minVelocity) / (maxVelocity - minVelocity) * numStates)));
   }
   public void calculate(int actionIndex) {
       //formula 
       velocity += Action[actionIndex] * 0.001 + Math.cos(3 * position) * (-0.0025);
       position += velocity;
      
       if (position > maxPosition)
           position = maxPosition;
       if (position < minPosition)
           position = minPosition;
       if (velocity > maxVelocity)
           velocity = maxVelocity;
       if (velocity < minVelocity)
           velocity = minVelocity;
   }
   public void initQtable(){
       for (int i = 0; i < numStates; i++) {
           for (int j = 0; j < numStates; j++) {
               int key = i * numStates + j;
               // initialize q table
               q.put(key, new double[] { 0, 0, 0 });
           }
       }
   }
   // maximize the total reward
   public MountainCarProblem(double alpha, double epsilon, double gamma, double decay, int numIter) {
      
       initQtable();
       int steps;

       for (int iter = 0; iter < numIter; iter++) { //loop for each episode
           State s = new State(0,0,0);
           // double[] qActions = s.getQS(s);
           // reset velocity and position for each iteration
           position = -0.5; //inital state //reset for every iteration
           velocity = 0.0; //inital state
           steps = 0;
           while (!s.isTerminal(position, maxPosition)) {  //loop for each step of episode
              
               steps++; //increment step
               int currentState = discretizedPosition(position) * numStates + discretizedVelocity(velocity);
               int actionIndex = getEpsilonGreedy(currentState, epsilon); //get epsilon greedy
               calculate(actionIndex); //take action A observer R and S aka our position and velocity
               int newState = discretizedPosition(position) * numStates + discretizedVelocity(velocity);
               // reward = 0;
               if (s.isTerminal(position, maxPosition)) { //if terminal state reached
                   //if no more steps to take add it to steps
                   successes++; 
                   stepsToGoal.add(steps); //add the amount of steps to list
                  // break; //break termination condition reached for that run
               } else {
                   //For every time step:
                   //enforce a negative reward for more steps taken
                   reward = -1;
               }

               
               double[] qValues = q.get(newState);
               double maxQ = qValues[0]; // Initialize maxQ with the first element
               for (int i = 1; i < qValues.length; i++) { //get maxQ
                   if (qValues[i] > maxQ) {
                       maxQ = qValues[i]; // Update maxQ if a larger value is found
                   }
               }
               //chose A' as a function of qˆ(S0,·,w) (e.g., "-greedy)
               //w <- w + a[rR + gamma*maxQ]
               //q.get(currentState)[actionIndex] += alpha * (reward + gamma * maxQ - q.get(currentState)[actionIndex]);
               q.get(currentState)[actionIndex] = (1-alpha) * q.get(currentState)[actionIndex] + alpha * (reward + gamma * maxQ);
               
               
               //s = s'
               //a = a'
               //System.out.println(position);
               alpha *= decay;
               epsilon *= decay;
           }
       }

       totalAttempts = numIter; //kinda useless.
   }
   // summarize progression results
   public void summarize() {
       if (stepsToGoal.isEmpty()) {
           // goal not met
           return;
       }

       int totalSteps = 0;
       for (int steps : stepsToGoal) { // get total number of steps
           totalSteps += steps;
       }
       double averageSteps = (double) totalSteps / stepsToGoal.size();
       System.out.println("Average Steps " + averageSteps);
       System.out.println("Steps in first success: " + stepsToGoal.get(0)); // showing learning over time
       System.out.println("Steps in last success: " + stepsToGoal.get(stepsToGoal.size() - 1));
   }
   public void exportDataToCSV(String filename) {
       try (FileWriter writer = new FileWriter(filename)) {
           for (Integer steps : stepsToGoal) {
               writer.append(steps.toString());
               writer.append("\n");
           }
       } catch (IOException e) {
           e.printStackTrace();
       }
   }
   
   public static void main(String[] args) {
       double epsilon = 0.1;
       double gamma = 0.99;
       double alpha = 0.1;
       double decay = .9999999;
       int numIter = 10000;
       
       MountainCarProblem MCP = new MountainCarProblem(alpha, epsilon, gamma, decay, numIter);
       MCP.summarize();
       MCP.exportDataToCSV("stepsToGoal.csv");

   }
}
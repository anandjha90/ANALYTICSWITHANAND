# Creating a database
CREATE DATABASE IF NOT EXISTS UnderstandingWindowFunctions;


# Using the created database 
USE UnderstandingWindowFunctions;

# Creating a table named as salestable
CREATE TABLE IF NOT EXISTS sales(
	sales_id INT PRIMARY KEY, 
    sales_person_name VARCHAR(250) NOT NULL,
    product_name VARCHAR(100) NOT NULL,
    location VARCHAR(100) NOT NULL,
    quantity_sold INT NOT NULL, 
    amount decimal(10,2) NOT NULL
);

# Inserting values into the table
INSERT INTO sales (sales_id, sales_person_name, product_name, location, quantity_sold, amount) VALUES
(1, 'Rajesh Sharma', 'Vadapav', 'Maharashtra', 30, 1500.00),
(2, 'Anjali Mehta', 'Vadapav', 'Gujarat', 25, 1250.00),
(3, 'Suresh Patil', 'Vadapav', 'Madhya Pradesh', 40, 2000.00),
(4, 'Priya Kumar', 'Vadapav', 'Rajasthan', 20, 1000.00),
(5, 'Manoj Gupta', 'Vadapav', 'Karnataka', 35, 1750.00),
(6, 'Rohit Singh', 'Vadapav', 'Uttar Pradesh', 50, 2500.00),
(7, 'Sunita Yadav', 'Vadapav', 'Punjab', 45, 2250.00),
(8, 'Vijay Deshmukh', 'Vadapav', 'Maharashtra', 60, 3000.00),
(9, 'Neha Verma', 'Vadapav', 'Tamil Nadu', 55, 2750.00),
(10, 'Karan Patel', 'Vadapav', 'Gujarat', 70, 3500.00),
(11, 'Arjun Reddy', 'Vadapav', 'Andhra Pradesh', 80, 4000.00),
(12, 'Nikita Jain', 'Vadapav', 'Delhi', 65, 3250.00),
(13, 'Vikas Malhotra', 'Vadapav', 'Haryana', 30, 1500.00),
(14, 'Shruti Rao', 'Vadapav', 'Telangana', 40, 2000.00),
(15, 'Akash Pandey', 'Vadapav', 'Uttar Pradesh', 45, 2250.00),
(16, 'Meera Shah', 'Vadapav', 'Maharashtra', 50, 2500.00),
(17, 'Ravi Sinha', 'Vadapav', 'Bihar', 35, 1750.00),
(18, 'Divya Kapoor', 'Vadapav', 'Punjab', 25, 1250.00),
(19, 'Amit Khanna', 'Vadapav', 'West Bengal', 60, 3000.00),
(20, 'Simran Kaur', 'Vadapav', 'Himachal Pradesh', 55, 2750.00),
(21, 'Deepak Bhatt', 'Vadapav', 'Uttarakhand', 20, 1000.00),
(22, 'Ayesha Khan', 'Vadapav', 'Maharashtra', 70, 3500.00),
(23, 'Pankaj Mishra', 'Vadapav', 'Odisha', 80, 4000.00),
(24, 'Ritika Joshi', 'Vadapav', 'Kerala', 65, 3250.00),
(25, 'Shivani Desai', 'Vadapav', 'Goa', 30, 1500.00),
(26, 'Abhinav Choudhary', 'Vadapav', 'Rajasthan', 40, 2000.00),
(27, 'Harsh Agarwal', 'Vadapav', 'Madhya Pradesh', 45, 2250.00),
(28, 'Tanya Srivastava', 'Vadapav', 'Uttar Pradesh', 50, 2500.00),
(29, 'Ramesh Joshi', 'Vadapav', 'Haryana', 35, 1750.00),
(30, 'Sneha Saxena', 'Vadapav', 'Karnataka', 25, 1250.00),
(31, 'Gaurav Nair', 'Vadapav', 'Tamil Nadu', 60, 3000.00),
(32, 'Anita Bhatia', 'Vadapav', 'Gujarat', 55, 2750.00),
(33, 'Puja Chatterjee', 'Vadapav', 'West Bengal', 70, 3500.00),
(34, 'Rahul Tripathi', 'Vadapav', 'Delhi', 80, 4000.00),
(35, 'Kavita Reddy', 'Vadapav', 'Andhra Pradesh', 65, 3250.00),
(36, 'Sanjay Iyer', 'Vadapav', 'Kerala', 30, 1500.00),
(37, 'Vidya Pillai', 'Vadapav', 'Karnataka', 40, 2000.00),
(38, 'Dinesh Chauhan', 'Vadapav', 'Punjab', 45, 2250.00),
(39, 'Rajiv Kapoor', 'Vadapav', 'Himachal Pradesh', 50, 2500.00),
(40, 'Mona Sharma', 'Vadapav', 'Uttarakhand', 35, 1750.00),
(41, 'Rahul Yadav', 'Vadapav', 'Bihar', 25, 1250.00),
(42, 'Ishita Gupta', 'Vadapav', 'Madhya Pradesh', 60, 3000.00),
(43, 'Nitin Shukla', 'Vadapav', 'Maharashtra', 55, 2750.00),
(44, 'Veena Singh', 'Vadapav', 'Rajasthan', 70, 3500.00),
(45, 'Ashok Nair', 'Vadapav', 'Tamil Nadu', 80, 4000.00),
(46, 'Rohini Kulkarni', 'Vadapav', 'Karnataka', 65, 3250.00),
(47, 'Shubham Rao', 'Vadapav', 'Telangana', 30, 1500.00),
(48, 'Nisha Patil', 'Vadapav', 'Maharashtra', 40, 2000.00),
(49, 'Keshav Sinha', 'Vadapav', 'Uttar Pradesh', 45, 2250.00),
(50, 'Payal Chauhan', 'Vadapav', 'Haryana', 50, 2500.00),
(51, 'Vikram Sharma', 'Samosa', 'Maharashtra', 30, 600.00),
(52, 'Pooja Mehta', 'Samosa', 'Gujarat', 25, 500.00),
(53, 'Sanjay Patil', 'Samosa', 'Madhya Pradesh', 40, 800.00),
(54, 'Deepika Kumar', 'Samosa', 'Rajasthan', 20, 400.00),
(55, 'Ankit Gupta', 'Samosa', 'Karnataka', 35, 700.00),
(56, 'Vivek Singh', 'Samosa', 'Uttar Pradesh', 50, 1000.00),
(57, 'Nidhi Yadav', 'Samosa', 'Punjab', 45, 900.00),
(58, 'Rakesh Deshmukh', 'Samosa', 'Maharashtra', 60, 1200.00),
(59, 'Seema Verma', 'Samosa', 'Tamil Nadu', 55, 1100.00),
(60, 'Abhay Patel', 'Samosa', 'Gujarat', 70, 1400.00),
(61, 'Vishal Reddy', 'Samosa', 'Andhra Pradesh', 80, 1600.00),
(62, 'Priyanka Jain', 'Samosa', 'Delhi', 65, 1300.00),
(63, 'Rahul Malhotra', 'Samosa', 'Haryana', 30, 600.00),
(64, 'Kriti Rao', 'Samosa', 'Telangana', 40, 800.00),
(65, 'Vishnu Pandey', 'Samosa', 'Uttar Pradesh', 45, 900.00),
(66, 'Radhika Shah', 'Samosa', 'Maharashtra', 50, 1000.00),
(67, 'Manish Sinha', 'Samosa', 'Bihar', 35, 700.00),
(68, 'Juhi Kapoor', 'Samosa', 'Punjab', 25, 500.00),
(69, 'Ashish Khanna', 'Samosa', 'West Bengal', 60, 1200.00),
(70, 'Ritu Kaur', 'Samosa', 'Himachal Pradesh', 55, 1100.00),
(71, 'Deepak Bhatt', 'Samosa', 'Uttarakhand', 20, 400.00),
(72, 'Alok Khan', 'Samosa', 'Maharashtra', 70, 1400.00),
(73, 'Harshit Mishra', 'Samosa', 'Odisha', 80, 1600.00),
(74, 'Lavanya Joshi', 'Samosa', 'Kerala', 65, 1300.00),
(75, 'Nikhil Desai', 'Samosa', 'Goa', 30, 600.00),
(76, 'Ishaan Choudhary', 'Samosa', 'Rajasthan', 40, 800.00),
(77, 'Prateek Agarwal', 'Samosa', 'Madhya Pradesh', 45, 900.00),
(78, 'Sneha Srivastava', 'Samosa', 'Uttar Pradesh', 50, 1000.00),
(79, 'Sumit Joshi', 'Samosa', 'Haryana', 35, 700.00),
(80, 'Megha Saxena', 'Samosa', 'Karnataka', 25, 500.00),
(81, 'Kunal Nair', 'Samosa', 'Tamil Nadu', 60, 1200.00),
(82, 'Tanvi Bhatia', 'Samosa', 'Gujarat', 55, 1100.00),
(83, 'Shalini Chatterjee', 'Samosa', 'West Bengal', 70, 1400.00),
(84, 'Naveen Tripathi', 'Samosa', 'Delhi', 80, 1600.00),
(85, 'Anusha Reddy', 'Samosa', 'Andhra Pradesh', 65, 1300.00),
(86, 'Ganesh Iyer', 'Samosa', 'Kerala', 30, 600.00),
(87, 'Swati Pillai', 'Samosa', 'Karnataka', 40, 800.00),
(88, 'Mohan Chauhan', 'Samosa', 'Punjab', 45, 900.00),
(89, 'Rohit Kapoor', 'Samosa', 'Himachal Pradesh', 50, 1200.00),
(90, 'Shalini Sharma', 'Samosa', 'Uttarakhand', 35, 701.00),
(91, 'Amit Yadav', 'Samosa', 'Bihar', 25, 502.00),
(92, 'Priya Gupta', 'Samosa', 'Madhya Pradesh', 60, 1210.00),
(93, 'Rajat Shukla', 'Samosa', 'Maharashtra', 55, 1110.00),
(94, 'Nikita Singh', 'Samosa', 'Rajasthan', 70, 1420.00),
(95, 'Siddharth Nair', 'Samosa', 'Tamil Nadu', 80, 1633.00),
(96, 'Pallavi Kulkarni', 'Samosa', 'Karnataka', 65, 1333.00),
(97, 'Varun Rao', 'Samosa', 'Telangana', 30, 601.00),
(98, 'Sneha Patil', 'Samosa', 'Maharashtra', 40, 807.00),
(99, 'Raj Sinha', 'Samosa', 'Uttar Pradesh', 45, 902.00),
(100, 'Komal Chauhan', 'Samosa', 'Haryana', 50, 1003.00),
(101, 'Rakesh Sharma', 'Dosa', 'Maharashtra', 20, 402.10),
(102, 'Aarti Mehta', 'Pani Puri', 'Gujarat', 35, 525.00),
(103, 'Siddharth Patil', 'Jalebi', 'Madhya Pradesh', 40, 803.00),
(104, 'Priya Kumar', 'Dosa', 'Rajasthan', 25, 509.00),
(105, 'Rohit Gupta', 'Pani Puri', 'Karnataka', 50, 70.00),
(106, 'Vikram Singh', 'Jalebi', 'Uttar Pradesh', 30, 610.00),
(107, 'Sunil Yadav', 'Dosa', 'Punjab', 60, 12.00),
(108, 'Nitin Deshmukh', 'Pani Puri', 'Maharashtra', 45, 6.00),
(109, 'Seema Verma', 'Jalebi', 'Tamil Nadu', 55, 1.00),
(110, 'Ankit Patel', 'Dosa', 'Gujarat', 70, 14.00),
(111, 'Praveen Reddy', 'Pani Puri', 'Andhra Pradesh', 80, 1.00),
(112, 'Nikita Jain', 'Jalebi', 'Delhi', 65, 130.00),
(113, 'Rakesh Malhotra', 'Dosa', 'Haryana', 30, 60.00),
(114, 'Pooja Rao', 'Pani Puri', 'Telangana', 40, 70.00),
(115, 'Ravi Pandey', 'Jalebi', 'Uttar Pradesh', 45, 91.00),
(116, 'Nidhi Shah', 'Dosa', 'Maharashtra', 50, 1001.00),
(117, 'Raj Sinha', 'Pani Puri', 'Bihar', 35, 525.00),
(118, 'Anjali Kapoor', 'Jalebi', 'Punjab', 25, 500.00),
(119, 'Amit Khanna', 'Dosa', 'West Bengal', 60, 1200.00),
(120, 'Ritu Kaur', 'Pani Puri', 'Himachal Pradesh', 55, 825.00),
(121, 'Deepak Bhatt', 'Jalebi', 'Uttarakhand', 20, 400.00),
(122, 'Ajay Khan', 'Dosa', 'Maharashtra', 70, 1400.00),
(123, 'Pankaj Mishra', 'Pani Puri', 'Odisha', 80, 1200.00),
(124, 'Lavanya Joshi', 'Jalebi', 'Kerala', 65, 1300.00),
(125, 'Ishaan Desai', 'Dosa', 'Goa', 30, 600.00),
(126, 'Ankit Choudhary', 'Pani Puri', 'Rajasthan', 40, 600.00),
(127, 'Ravi Agarwal', 'Jalebi', 'Madhya Pradesh', 45, 900.00),
(128, 'Sonal Srivastava', 'Dosa', 'Uttar Pradesh', 50, 1000.00),
(129, 'Sumit Joshi', 'Pani Puri', 'Haryana', 35, 525.00),
(130, 'Megha Saxena', 'Jalebi', 'Karnataka', 25, 500.00),
(131, 'Gaurav Nair', 'Dosa', 'Tamil Nadu', 60, 1200.00),
(132, 'Anita Bhatia', 'Pani Puri', 'Gujarat', 55, 825.00),
(133, 'Puja Chatterjee', 'Jalebi', 'West Bengal', 70, 1400.00),
(134, 'Rohit Tripathi', 'Dosa', 'Delhi', 80, 1600.00),
(135, 'Kavita Reddy', 'Pani Puri', 'Andhra Pradesh', 65, 975.00),
(136, 'Ganesh Iyer', 'Jalebi', 'Kerala', 30, 600.00),
(137, 'Vidya Pillai', 'Dosa', 'Karnataka', 40, 800.00),
(138, 'Dinesh Chauhan', 'Pani Puri', 'Punjab', 45, 675.00),
(139, 'Rajiv Kapoor', 'Jalebi', 'Himachal Pradesh', 50, 1000.00),
(140, 'Mona Sharma', 'Dosa', 'Uttarakhand', 35, 700.00),
(141, 'Rahul Yadav', 'Pani Puri', 'Bihar', 25, 375.00),
(142, 'Priya Gupta', 'Jalebi', 'Madhya Pradesh', 60, 1200.00),
(143, 'Rajat Shukla', 'Dosa', 'Maharashtra', 55, 1100.00),
(144, 'Nikita Singh', 'Pani Puri', 'Rajasthan', 70, 1050.00),
(145, 'Siddharth Nair', 'Jalebi', 'Tamil Nadu', 80, 1600.00),
(146, 'Pallavi Kulkarni', 'Dosa', 'Karnataka', 65, 1300.00),
(147, 'Varun Rao', 'Pani Puri', 'Telangana', 30, 450.00),
(148, 'Sneha Patil', 'Jalebi', 'Maharashtra', 40, 800.00),
(149, 'Keshav Sinha', 'Dosa', 'Uttar Pradesh', 45, 900.00),
(150, 'Komal Chauhan', 'Pani Puri', 'Haryana', 50, 750.00),
(151, 'Sumit Joshi', 'Pani Puri', 'Haryana', 35, 5233.00),
(152, 'Megha Saxena', 'Jalebi', 'Karnataka', 25, 521.00),
(153, 'Gaurav Nair', 'Dosa', 'Tamil Nadu', 60, 123.00),
(154, 'Anita Bhatia', 'Pani Puri', 'Gujarat', 55, 823.00),
(155, 'Puja Chatterjee', 'Jalebi', 'West Bengal', 70, 142.00),
(156, 'Rohit Tripathi', 'Dosa', 'Delhi', 80, 164.00),
(157, 'Kavita Reddy', 'Pani Puri', 'Andhra Pradesh', 65, 1745.00),
(158, 'Ganesh Iyer', 'Jalebi', 'Kerala', 30, 1223.00),
(159, 'Vidya Pillai', 'Dosa', 'Karnataka', 40, 81.00),
(160, 'Dinesh Chauhan', 'Pani Puri', 'Punjab', 45, 67.00),
(161, 'Rajiv Kapoor', 'Jalebi', 'Himachal Pradesh', 50, 10.00),
(162, 'Mona Sharma', 'Dosa', 'Uttarakhand', 35, 99.00),
(163, 'Rahul Yadav', 'Pani Puri', 'Bihar', 25, 382.00),
(164, 'Priya Gupta', 'Jalebi', 'Madhya Pradesh', 60, 140.00),
(165, 'Rajat Shukla', 'Dosa', 'Maharashtra', 55, 123.00),
(166, 'Nikita Singh', 'Pani Puri', 'Rajasthan', 70, 11.00),
(167, 'Siddharth Nair', 'Jalebi', 'Tamil Nadu', 80, 1610.00),
(168, 'Pallavi Kulkarni', 'Dosa', 'Karnataka', 65, 1320.00),
(169, 'Varun Rao', 'Pani Puri', 'Telangana', 30, 400.00),
(170, 'Sneha Patil', 'Jalebi', 'Maharashtra', 40, 81.00),
(171, 'Keshav Sinha', 'Dosa', 'Uttar Pradesh', 45, 91.00),
(172, 'Komal Chauhan', 'Pani Puri', 'Haryana', 50, 75.00),
(173, 'Karan Shah', 'Jalebi', 'Madhya Pradesh', 60, 140.00),
(174, 'Karan Shah', 'Dosa', 'Maharashtra', 55, 123.00),
(175, 'Karan Shah', 'Pani Puri', 'Rajasthan', 70, 11.00),
(176, 'Karan Shah', 'Jalebi', 'Tamil Nadu', 80, 1610.00),
(177, 'Karan Shah', 'Dosa', 'Karnataka', 65, 1320.00),
(178, 'Karan Shah', 'Pani Puri', 'Telangana', 30, 400.00),
(179, 'Karan Shah', 'Jalebi', 'Maharashtra', 40, 81.00),
(180, 'Varun Rao', 'Dosa', 'Uttar Pradesh', 45, 91.00),
(181, 'Varun Rao', 'Pani Puri', 'Haryana', 50, 75.00),
(182, 'Rajesh Sharma', 'Jalebi', 'Madhya Pradesh', 60, 140.00),
(183, 'Rajesh Sharma', 'Dosa', 'Maharashtra', 55, 123.00),
(184, 'Rajesh Sharma', 'Pani Puri', 'Rajasthan', 70, 11.00),
(185, 'Varun Rao', 'Jalebi', 'Tamil Nadu', 80, 1610.00),
(186, 'Sneha Patil', 'Dosa', 'Karnataka', 65, 1320.00),
(187, 'Sneha Patil', 'Pani Puri', 'Telangana', 30, 400.00),
(188, 'Raj Sinha', 'Jalebi', 'Maharashtra', 40, 81.00),
(189, 'Komal Chauhan', 'Dosa', 'Uttar Pradesh', 45, 91.00),
(190, 'Varun Rao', 'Pani Puri', 'Haryana', 50, 75.00);

/*----------------------------------------------------------------------------------------------------------*/
# LET US BEGIN WITH THE SESSION
/*----------------------------------------------------------------------------------------------------------*/

# Question No 1
/*
Based on the above sales table, you must find the total sales amounts per location. 
*/

#Solution


# with window functions



# Question No 2
/*
Find the total aaverage sales amount for Product Name.
Also, results must be by the total sales in descending order. Note we want only 2 decimal values in the amount columns
*/

# Solution 


# Solution using window functions
# SOLUTION


# Question No 3
/*
Based on the given sales table. 
Write an SQL query to retrive the total sales amount and the average sales amount per location per sales person.
Note that the amount must only have max of 2 decimal points. 
*/
# SOLUTION


/*------------------------------------------------------------------------------------------------------*/
# Section 2 - Ranking Functions
/*
	Write an SQL query in such a way that we can find out the which are the top three sales person with the highest sales
    from the Maharashtra location.
*/

# Solution 


/*
# The above functions are one of the way, but they are not the ultimate solutions 
# or they are not the optimum solution for the given problem.
# For this, therefore we need to use analytical windowing function.
# There are three popular windowing functions such as RANK(), ROW(), DENSERANK().
*/

# Ranking functions
# Function 1 - ROW_NUMBER()
# Syntax - ROW_NUMBER() OVER()

/*
	SAME QUESTION
	Write an SQL query to rank the sales or the sales person
*/
# SOLUTION

/*
	SAME QUESTION
	Write an SQL query in such a way that we can find out the which are the top three sales person with the highest sales
    from the Maharashtra location.
*/
# SOLUTION

/*
	Write an SQL query in such a way that we can find out the which are the top three sales person with the highest sales
*/
# SOLUTION


# SECTION 2
# Rank Function
/*
    RANK() : - 
		The rank function basically ranks the dataset based on the the given conditions
        The only difference here is that it gives the same rank to the same values present, and skips the 
        total number of consecutive counts of the same value. 
        For example :- If there are two people with the highest sales present in the dataset 
        It will rank both of them as 1, and not as 1 and 2. After that the 3rd value that is the 2nd highest 
        student will be assigned the value of 3 and not 2.
*/

/*
	Write an SQL query to rank the sales of the sales person
*/

# First using ROW_NUMBER()

# Now using RANK()


/*
	SAME QUESTION
	Write an SQL query in such a way that we can find out the which are the top three sales person with the highest sales
    from the Maharashtra location.
*/

# First using ROW_NUMBER()

# using RANK() function


# SECTION 3
# Dense RANK()
# DENSE_RANK() :- It is same as rank, but with only some minor differences. 
# Dence Rank solves the issue of skiping the rank for the value if any previous 
# consecutive same rank has occured.
/*
	Write an SQL query to rank the sales of the sales person
*/

# First using ROW_NUMBER()

# Now using RANK()

# Now Using Dense_RANK()


# PRACTICE QUESTIONS
# QUESTION 1
/*
Write a query to rank all salespersons based on their total sales amount for each location. 
If two salespersons have the same total sales amount in a location, they should receive the same rank, 
and subsequent ranks should skip the appropriate numbers. Which function would you use and why?
*/
# Solution



# QUESTION 2
/*
Write a query to find the top 3 sales amounts for each product. 
Each entry should be uniquely identified, even if multiple sales have the same amount. 
Which function should you use to achieve this?
*/
# Solution


# QUESTION 3
/*
Write a query to find the top 3 sales amounts for each product and for each location. 
Note that the cases where the sales amount is same must be given the same rank. There must be no skipping of the 
ranks.
*/
# Solution




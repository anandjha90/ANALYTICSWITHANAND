-- set up roles
CREATE OR REPLACE ROLE ANALYST;
CREATE OR REPLACE ROLE TESTER;
CREATE OR REPLACE ROLE DEVELOPER;

CREATE OR REPLACE ROLE DESIGNER;
-- Set Up User

-- For TONY User
CREATE OR REPLACE USER user_designer
PASSWORD = 'Designer123!'
DEFAULT_ROLE = DESIGNER
DEFAULT_WAREHOUSE = DEMO_WAREHOUSE
MUST_CHANGE_PASSWORD = TRUE;  -- User will be prompted to change the password on first login

-- For TONY User
CREATE OR REPLACE USER user_tony
PASSWORD = 'Tony123!'
DEFAULT_ROLE = ANALYST
DEFAULT_WAREHOUSE = DEMO_WAREHOUSE
MUST_CHANGE_PASSWORD = TRUE;  -- User will be prompted to change the password on first login

-- For STEVE User
CREATE OR REPLACE USER user_steve
PASSWORD = 'Steve123!'
DEFAULT_ROLE = TESTER
DEFAULT_WAREHOUSE = DEMO_WAREHOUSE
MUST_CHANGE_PASSWORD = TRUE;

-- -- For BRUCE User
CREATE OR REPLACE USER user_bruce
PASSWORD = 'Bruce123!'
DEFAULT_ROLE = DEVELOPER
DEFAULT_WAREHOUSE = DEMO_WAREHOUSE
MUST_CHANGE_PASSWORD = TRUE;

grant role ANALYST to user user_tony;
grant role DEVELOPER to user user_bruce;
grant role TESTER to user user_steve;
grant role DESIGNER to user user_designer;

GRANT USAGE ON WAREHOUSE DEMO_WAREHOUSE TO ROLE SECURITYADMIN;

--  Granting Necessary Access to the Designer role
GRANT USAGE ON WAREHOUSE DEMO_WAREHOUSE TO ROLE DESIGNER;
GRANT USAGE ON DATABASE DEMO_DATABASE TO ROLE DESIGNER;
GRANT ALL ON SCHEMA DEMO_SCHEMA TO ROLE DESIGNER;
GRANT ALL PRIVILEGES ON SCHEMA DEMO_SCHEMA TO ROLE DESIGNER;
GRANT SELECT ON ALL TABLES IN SCHEMA DEMO_SCHEMA TO ROLE DESIGNER;


-- Granting Necessary Access to the DEVELOPER role
GRANT USAGE ON WAREHOUSE DEMO_WAREHOUSE TO ROLE DEVELOPER;
GRANT USAGE ON DATABASE DEMO_DATABASE TO ROLE DEVELOPER;
GRANT ALL ON SCHEMA DEMO_SCHEMA TO ROLE DEVELOPER;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA DEMO_SCHEMA TO ROLE DEVELOPER;
GRANT ALL PRIVILEGES ON SCHEMA DEMO_SCHEMA TO ROLE DEVELOPER;
GRANT INSERT, UPDATE, DELETE ON FUTURE TABLES IN SCHEMA DEMO_SCHEMA TO ROLE DEVELOPER;


-- Granting Necessary Access to the ANALYST role
GRANT USAGE ON WAREHOUSE DEMO_WAREHOUSE TO ROLE ANALYST;
GRANT USAGE ON DATABASE DEMO_DATABASE TO ROLE ANALYST;
GRANT ALL ON SCHEMA DEMO_SCHEMA TO ROLE ANALYST;
GRANT ALL PRIVILEGES ON SCHEMA DEMO_SCHEMA TO ROLE ANALYST;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA DEMO_SCHEMA TO ROLE ANALYST;

-- Granting Necessary Access to the TESTER role
GRANT USAGE ON WAREHOUSE DEMO_WAREHOUSE TO ROLE TESTER;
GRANT USAGE ON DATABASE DEMO_DATABASE TO ROLE TESTER;
GRANT ALL PRIVILEGES ON SCHEMA DEMO_SCHEMA TO ROLE TESTER;
GRANT SELECT ON ALL TABLES IN SCHEMA DEMO_SCHEMA TO ROLE TESTER;


-- SHow grants
SHOW GRANTS ON DATABASE DEMO_DATABASE;
SHOW GRANTS ON SCHEMA DEMO_SCHEMA;

SHOW GRANTS TO USER user_tony;
SHOW GRANTS TO USER user_steve;
SHOW GRANTS TO USER user_bruce;

SHOW GRANTS TO ROLE DEVELOPER;
SHOW GRANTS TO ROLE TESTER;
SHOW GRANTS TO ROLE ANALYST;

DROP TABLE CREDIT_CARD_CUSTOMER;
--Lets create this table under developer role
CREATE OR REPLACE TABLE CREDIT_CARD_CUSTOMER
(
   id number,
   first_name string, 
   Last_name string,
   DoB string,
   creditcard string,
   PAN string,
   city string,
   country string
 );

INSERT INTO CREDIT_CARD_CUSTOMER VALUES
(1, 'Vishal', 'Kaushal','1990-01-01',468453451278,'ABCD','Delhi', 'India'),
(2, 'Rohit', 'Sharma','1991-01-01',346129428914,'CDEF','Mumbai', 'India'),
(3, 'Virat', 'Kohli','1992-01-01',456237318920,'DEFG','Bangalore', 'India'),
(4, 'Shami', 'Ahmad','1993-01-01',451672893013,'HIJK','Lucknow','India'),
(5, 'Jasprit', 'Bumrah','1994-01-01',246871249124,'LMNOP','Mumbai','India'),
(6, 'Axar' ,'Patel','1995-01-01',412916498123,'QRSTU','Ahemdabad','India');


SELECT * FROM CREDIT_CARD_CUSTOMER;

-- Set up masking policy--

CREATE OR REPLACE TABLE CREDIT_CARD_CUSTOMER
(
   id number,
   first_name string, 
   Last_name string,
   DoB string,
   creditcard string,
   PAN string,
   city string,
   country string
 );

INSERT INTO CREDIT_CARD_CUSTOMER VALUES
(1, 'Vishal', 'Kaushal','1990-01-01',468453451278,'ABCD','Delhi', 'India'),
(2, 'Rohit', 'Sharma','1991-01-01',346129428914,'CDEF','Mumbai', 'India'),
(3, 'Virat', 'Kohli','1992-01-01',456237318920,'DEFG','Bangalore', 'India'),
(4, 'Shami', 'Ahmad','1993-01-01',451672893013,'HIJK','Lucknow','India'),
(5, 'Jasprit', 'Bumrah','1994-01-01',246871249124,'LMNOP','Mumbai','India'),
(6, 'Axar' ,'Patel','1995-01-01',412916498123,'QRSTU','Ahemdabad','India');


SELECT * FROM CREDIT_CARD_CUSTOMER;

CREATE OR REPLACE MASKING POLICY MASK_CREDT_CARD AS (creditcard STRING) 
RETURNS STRING -> 
CASE 
   WHEN CURRENT_ROLE() IN ('DEVELOPER','ACCOUNTADMIN') THEN creditcard
   WHEN CURRENT_ROLE() IN ('ANALYST') THEN REGEXP_REPLACE(creditcard, '^.{7}', '*******')
   WHEN CURRENT_ROLE() IN ('TESTER') THEN '**********'
   ELSE '***CAN NOT BE SEEN***'
END;

CREATE OR REPLACE MASKING POLICY MASK_PAN_CARD AS (PAN STRING) RETURNS STRING ->
  CASE 
    WHEN CURRENT_ROLE() IN ('DEVELOPER','ACCOUNTADMIN') THEN PAN
    WHEN CURRENT_ROLE() IN ('ANALYST') THEN REPLACE(PAN,SUBSTR(PAN,1,4),'$$$$')
    WHEN CURRENT_ROLE() IN ('TESTER') THEN '**********'
    ELSE '***CAN NOT BEEN SEEN***'
END;

-- FOR AN EXISTING TABLE ON VIEW, EXECUTE THE FOLLOWING STATEMENT--

ALTER TABLE IF EXISTS CREDIT_CARD_CUSTOMER MODIFY COLUMN creditcard SET MASKING POLICY MASK_CREDT_CARD;
ALTER TABLE IF EXISTS CREDIT_CARD_CUSTOMER MODIFY COLUMN PAN SET MASKING POLICY MASK_PAN_CARD;

DESC TABLE CREDIT_CARD_CUSTOMER;

GRANT ALL PRIVILEGES ON TABLE CREDIT_CARD_CUSTOMER TO ROLE ACCOUNTADMIN;
GRANT ALL PRIVILEGES ON TABLE CREDIT_CARD_CUSTOMER TO ROLE TESTER;
GRANT ALL PRIVILEGES ON TABLE CREDIT_CARD_CUSTOMER TO ROLE ACCOUNTADMIN;
GRANT ALL PRIVILEGES ON TABLE CREDIT_CARD_CUSTOMER TO ROLE ANALYST;

--How we can remove the masking policy--
| alter table if exists CREDIT_CARD_CUSTOMER modify column creditcard unset masking policy;
  alter table if exists CREDIT_CARD_CUSTOMER modify column PAN unset masking policy;



-- FOR AN EXISTING TABLE ON VIEW, EXECUTE THE FOLLOWING STATEMENT--

ALTER TABLE IF EXISTS CREDIT_CARD_CUSTOMER MODIFY COLUMN creditcard SET MASKING POLICY P1_MASKING;
ALTER TABLE IF EXISTS CREDIT_CARD_CUSTOMER MODIFY COLUMN PAN SET MASKING POLICY P2_MASKING;

DESC TABLE CREDIT_CARD_CUSTOMER;

-- Validating policies

USE ROLE ACCOUNTADMIN;
SELECT * FROM DEMO_DATABASE.DEMO_SCHEMA.CREDIT_CARD_CUSTOMER;

USE ROLE ANALYST;
SELECT * FROM DEMO_DATABASE.DEMO_SCHEMA.CREDIT_CARD_CUSTOMER;

USE ROLE DEVELOPER;
SELECT * FROM DEMO_DATABASE.DEMO_SCHEMA.CREDIT_CARD_CUSTOMER;


USE ROLE TESTER;
SELECT * FROM DEMO_DATABASE.DEMO_SCHEMA.CREDIT_CARD_CUSTOMER;





--How we can remove the masking policy--
| alter table if exists CREDIT_CARD_CUSTOMER modify column creditcard unset masking policy;
  alter table if exists CREDIT_CARD_CUSTOMER modify column PAN unset masking policy;

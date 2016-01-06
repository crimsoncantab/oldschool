<?
    //reqire common code
    require_once("inc/common.inc");
?>

<!DOCTYPE html 
     PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
          "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

<html xmlns="http://www.w3.org/1999/xhtml">
  <head>
    <title>Coach's Login</title>
  </head>
  <body>
    <form action="login2.php" method="post">
      <div align="center">
       Username:
        <input name="username" type="text" />
      </div>
      <br />
      <div align="center">
      Password:
        <input name="password" type="password" />
      </div>
      <br />
      <div align="center">
      <input type="submit" value="Submit" />
      </div>
    </form>

    <div>
      <a href="register.php">Register</a>
    </div>
    <div>
      <a href="index.php">Back to Main Page</a>
    </div>
  </body>
</html>

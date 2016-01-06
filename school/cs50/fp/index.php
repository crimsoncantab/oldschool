<?
    // require common code
    require_once("inc/common.inc");
?>

<!DOCTYPE html 
     PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
          "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

<html xmlns="http://www.w3.org/1999/xhtml">
  <head>
    <title>Crew Practice Signup</title>
  </head>
  <body>
    <form action="signup.php" method="get">
      <div align="center">
      Pick your coach:
        <select name="coach">
          <option selected>Select Coach
          <?
              //imports coaches to be listed in form
              $names = mysql_query("SELECT name FROM users");
              while ($row = mysql_fetch_array($names)) {
                  print("<option>" . $row["name"]);
              }
          ?>
        </select>
      </div>
      <div align="center">
        <input type="submit" value="Submit" />
      </div>
    </form>
    <div>
      <a href="login.php">Coach's link</a>
    </div>
  </body>
</html>

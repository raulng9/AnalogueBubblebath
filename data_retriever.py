f = open("students.txt", "r")


class Student:
  def __init__(self, name, listOfMarks):
    self.name = name
    self.listOfMarks = listOfMarks


listOfStudents = []

for student in f:
    studentData = student.split()
    listOfMarksForStudent = []
    for x in range (1,len(studentData)):
        listOfMarksForStudent.append(int(studentData[x]))
    newStudent = Student(student, listOfMarksForStudent)
    print(listOfMarksForStudent)
